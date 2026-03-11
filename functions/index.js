// Firebase Cloud Function for Eye Symptom Assistant API
// Adapted from server.js for serverless deployment with Llama classifiers

import { onRequest } from "firebase-functions/v2/https";
import { defineSecret } from "firebase-functions/params";
import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import crypto from "crypto";

// Define secrets (set via: firebase functions:secrets:set SECRET_NAME)
const azureEndpoint = defineSecret("AZURE_OPENAI_ENDPOINT");
const azureApiKey = defineSecret("AZURE_OPENAI_API_KEY");
const azureDeployment = defineSecret("AZURE_OPENAI_DEPLOYMENT");
const hfToken = defineSecret("HF_TOKEN");

const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: "1mb" }));

// Llama model for classifiers
const LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct";

// In-memory session store (Note: stateless in serverless, but works for single request)
const sessions = new Map();

// ============================================================================
// System Prompt
// ============================================================================

const ASSISTANT_SYSTEM_PROMPT = `
You are a friendly, responsible, ophthalmology-focused Eye Health Assistant.

SCOPE RESTRICTION (VERY IMPORTANT):
- You ONLY help with eye-related concerns (redness, pain, vision changes, dryness, irritation,
  discharge, floaters, light sensitivity, eyelid issues, trauma, etc.).
- If the user mentions symptoms NOT related to the eyes (e.g., headache alone, chest pain,
  stomach pain, fever, cough, dizziness without eye involvement):
    → Do NOT give medical advice.
    → Politely explain you only handle eye symptoms.
    → Ask whether anything is bothering their eyes.
- If the user insists on non-eye topics → repeat scope limitation.

BEHAVIOUR:
- If the user only greets → greet ONCE and briefly explain what you can help with.
- If the user types Snellen chart letters → estimate visual acuity + note it is only a screening test.
- For eye symptoms → ask clarifying questions:
  onset, one/both eyes, pain 0–10, vision changes, discharge, light sensitivity,
  contact lenses, trauma/chemical exposure, recent illness.
- After collecting enough info → provide:
  • 1–2 sentence summary
  • 2–3 common non-scary possible causes
  • simple self-care tips (never medications)
  • when to see an ophthalmologist
  • TRIAGE_JSON (see format below)

INTERACTION RULES (STRICT):
- NEVER provide causes or advice before completing clarifying questions.
- Ask up to 3 clarifying questions, one at a time:
    Q1 → wait → Q2 → wait → Q3 → wait → final response.
- If user asks to skip questions → skip to final response.
- If enough detail is already present → skip remaining questions and respond.
- Do NOT greet again after your first message.

RED-FLAG OVERRIDE:
If any of these appear:
- sudden vision loss or severe blur
- flashes + floaters + curtain/shadow
- severe eye pain with nausea/vomiting
- chemical splash or eye injury
- painful red eye in a contact-lens wearer
Then:
→ Skip remaining questions
→ Immediately give urgent/emergency guidance
→ Output TRIAGE_JSON accordingly

SAFETY RULES (HARD LIMITS):
1. Never diagnose or say "you have".
2. Never recommend medications, eye drops, dosages, treatments.
3. Never name medications.
4. Use phrases like "common possibilities include…".
5. Always end final response with:
   "This is general information only — not a diagnosis.
    See an ophthalmologist if symptoms persist or worry you."

OUTPUT FORMAT (MANDATORY):
After your explanation (max ~200 words), output:

TRIAGE_JSON: {
  "triage": "self-care" | "routine-ophthalmologist" | "urgent-24h" | "emergency-now",
  "red_flags_detected": ["..."],
  "mentions_meds": true | false
}

Do NOT add anything after this JSON.
Be warm, concise, and supportive.
`.trim();

// ============================================================================
// Helper Functions
// ============================================================================

function getOrCreateSession(sessionId) {
  let id = sessionId;
  if (!id || !sessions.has(id)) {
    id = id || crypto.randomUUID();
    sessions.set(id, {
      messages: [{ role: "system", content: ASSISTANT_SYSTEM_PROMPT }],
    });
  }
  return { id, session: sessions.get(id) };
}

function preCheckUserInput(text) {
  if (!text || typeof text !== "string") {
    return { blocked: true, safeReply: "I couldn't read your message. Please describe your eye concern." };
  }

  const lower = text.toLowerCase();
  const medicationPatterns = [" mg ", "mg/", "dosage", "dose", "prescribe", "prescription", "antibiotic", "steroid", "eye drops with", "drops of", "tablet", "pill"];

  if (medicationPatterns.some((p) => lower.includes(p))) {
    return { blocked: true, safeReply: "I'm not able to discuss medications or dosages. I can only give general information about eye symptoms." };
  }

  const selfHarmPatterns = ["kill myself", "suicide", "end my life", "hurt myself", "don't want to live"];
  if (selfHarmPatterns.some((p) => lower.includes(p))) {
    return {
      blocked: true,
      safeReply: "I'm really sorry that you're feeling this way. I'm not able to help with self-harm or suicide situations.\nPlease contact your local emergency number or a mental health helpline immediately.",
    };
  }

  return { blocked: false };
}

function detectLocalIntent(text) {
  const lower = text.trim().toLowerCase();
  const greetings = ["hi", "hello", "hey", "good morning", "good evening"];
  if (greetings.some((g) => lower.startsWith(g))) {
    return {
      matched: true,
      reply: "Hello! I'm your Eye Health Assistant.\nI can help with visual acuity tests and questions about eye symptoms.\nHow can I help your eyes today?",
    };
  }

  const closings = ["ok", "okay", "thanks", "thank you", "bye", "great", "good"];
  if (closings.some((c) => lower.startsWith(c))) {
    return { matched: true, reply: "You're welcome! If anything else bothers your eyes, feel free to ask!" };
  }

  return { matched: false };
}

function serializeConversation(messages) {
  return messages.filter((m) => m.role !== "system").map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
}

function parseTriageJsonFromText(text) {
  const result = { ok: false, triage: "unknown", red_flags_detected: [], mentions_meds: false };
  // Handle various formats: TRIAGE_JSON:, **TRIAGE_JSON**:, etc.
  const match = text.match(/\*{0,2}TRIAGE_JSON\*{0,2}:?\s*({[\s\S]*?})/i);
  if (!match) return result;

  try {
    const parsed = JSON.parse(match[1]);
    return { ok: true, triage: parsed.triage ?? "unknown", red_flags_detected: parsed.red_flags_detected ?? [], mentions_meds: parsed.mentions_meds ?? false };
  } catch {
    return result;
  }
}

function removeTriageJsonBlock(text) {
  // Handle various formats: TRIAGE_JSON:, **TRIAGE_JSON**:, ---TRIAGE_JSON, etc.
  return text
    .replace(/\*{0,2}TRIAGE_JSON\*{0,2}:?[\s\S]*$/i, "")
    .replace(/---+\s*$/g, "")
    .trim();
}

function containsMedicationLanguage(text) {
  const lower = text.toLowerCase();
  const patterns = [" mg ", "mg/", "dosage", "antibiotic", "steroid", "take ", "tablet", "pill"];
  return patterns.some((p) => lower.includes(p));
}

function looksLikeExamHallucination(text) {
  const lower = text.toLowerCase();
  const patterns = ["after examining your eye", "on your eye exam", "i have examined your eye", "i have seen your retina", "based on your scan", "your blood test shows", "i checked your test results"];
  return patterns.some((p) => lower.includes(p));
}

// ============================================================================
// API Callers
// ============================================================================

async function callAzureAssistant(messages, mode, config) {
  const url = `${config.endpoint}/openai/deployments/${config.deployment}/chat/completions?api-version=2024-02-15-preview`;
  const body = { messages, temperature: mode === "research" ? 0.5 : 0.2, max_tokens: 700 };

  const res = await fetch(url, {
    method: "POST",
    headers: { "api-key": config.apiKey, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error("[AZURE] Error:", res.status, err);
    throw new Error("Azure assistant call failed");
  }

  const json = await res.json();
  return json.choices?.[0]?.message?.content || "";
}

async function callLlamaClassifier(messages, maxTokens, token) {
  if (!token) throw new Error("HF_TOKEN missing");

  const res = await fetch("https://router.huggingface.co/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: LLAMA_MODEL, temperature: 0, max_tokens: maxTokens, messages }),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error("[LLAMA] Error:", res.status, err);
    throw new Error("Llama classifier call failed");
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || "";
  try {
    return JSON.parse(content);
  } catch {
    throw new Error("Llama JSON parse failed");
  }
}

// ============================================================================
// Classifiers
// ============================================================================

async function runPreRiskClassifier(messages, token) {
  try {
    if (!token) throw new Error("HF_TOKEN missing");

    const convo = serializeConversation(messages);
    const classifierMessages = [
      {
        role: "system",
        content:
          "You are a medical RISK classifier for an ophthalmology assistant.\n" +
          "Read the conversation and output ONLY valid JSON with exactly these fields:\n" +
          '{ "risk_level": "low" | "moderate" | "high" | "emergency", "request_type": "information" | "diagnosis" | "treatment" | "triage" | "self-harm" | "other", "requires_clarification": true | false, "prohibited": true | false, "notes": string }\n' +
          "Prohibited = true if the user explicitly requests diagnosis, medication, dosage, prescription, or self-harm instructions.\n" +
          "Do NOT include any extra text or explanations, only pure JSON.",
      },
      { role: "user", content: `CONVERSATION:\n${convo}` },
    ];

    const json = await callLlamaClassifier(classifierMessages, 220, token);
    return {
      risk_level: json.risk_level || "low",
      request_type: json.request_type || "other",
      requires_clarification: typeof json.requires_clarification === "boolean" ? json.requires_clarification : false,
      prohibited: !!json.prohibited,
      notes: json.notes || "",
    };
  } catch (err) {
    console.warn("[PRE-RISK] Failed, using safe defaults:", err.message);
    return { risk_level: "low", request_type: "other", requires_clarification: false, prohibited: false, notes: "" };
  }
}

async function runPostSafetyClassifier(userText, answerText, triageInfo, token) {
  try {
    if (!token) throw new Error("HF_TOKEN missing");

    const triageJson = JSON.stringify(triageInfo || {});
    const classifierMessages = [
      {
        role: "system",
        content:
          "You are a medical SAFETY classifier for an ophthalmology assistant.\n" +
          "Given a user symptom description, an assistant reply, and triage JSON, output ONLY valid JSON:\n" +
          '{ "safety": "ok" | "undertriage" | "overtriage" | "medication" | "diagnosis" | "hallucination_exam" | "other", "needs_block": true | false, "needs_upgrade_to_emergency": true | false, "reasons": [string] }\n' +
          "Do NOT output anything besides this JSON.",
      },
      { role: "user", content: `USER_SYMPTOMS:\n${userText}\n\nASSISTANT_REPLY:\n${answerText}\n\nTRIAGE_JSON:\n${triageJson}` },
    ];

    const json = await callLlamaClassifier(classifierMessages, 220, token);
    return {
      safety: json.safety || "ok",
      needs_block: !!json.needs_block,
      needs_upgrade_to_emergency: !!json.needs_upgrade_to_emergency,
      reasons: Array.isArray(json.reasons) ? json.reasons : [],
    };
  } catch (err) {
    console.warn("[POST-SAFETY] Failed, defaulting to ok:", err.message);
    return { safety: "ok", needs_block: false, needs_upgrade_to_emergency: false, reasons: [] };
  }
}

function validateAndCorrectTriage(triageInfo, preRisk, answerSafety) {
  let triage = triageInfo.triage || "unknown";

  if (triageInfo.red_flags_detected.length > 0 && triage === "self-care") {
    triage = "urgent-24h";
  }

  if (preRisk && (preRisk.risk_level === "high" || preRisk.risk_level === "emergency")) {
    if (triage === "self-care" || triage === "routine-ophthalmologist") {
      triage = preRisk.risk_level === "emergency" ? "emergency-now" : "urgent-24h";
    }
  }

  if (answerSafety && answerSafety.needs_upgrade_to_emergency) {
    triage = "emergency-now";
  }

  return triage;
}

/**
 * Returns a risk-aware fallback triage level when TRIAGE_JSON is missing.
 */
function getFallbackTriage(preRisk) {
  if (preRisk && preRisk.risk_level === "emergency") {
    return "emergency-now";
  }
  if (preRisk && preRisk.risk_level === "high") {
    return "urgent-24h";
  }
  return "routine-ophthalmologist";
}

// ============================================================================
// Main Chat Endpoint
// ============================================================================

app.post("/api/chat", async (req, res) => {
  try {
    let { sessionId, message, mode } = req.body;
    mode = mode || "normal";

    const userText = (message || "").toString().trim();
    const lowerUser = userText.toLowerCase();

    // Get secrets
    const AZURE_ENDPOINT = azureEndpoint.value()?.replace(/\/+$/, "");
    const AZURE_API_KEY = azureApiKey.value();
    const AZURE_DEPLOYMENT = azureDeployment.value() || "DeepSeek-V3.1";
    const HF_TOKEN = hfToken.value();

    const azureConfig = { endpoint: AZURE_ENDPOINT, apiKey: AZURE_API_KEY, deployment: AZURE_DEPLOYMENT };

    // Create or load session
    const { id: activeSessionId, session } = getOrCreateSession(sessionId);

    // Step 1: Local greeting/closing handling
    const localIntent = detectLocalIntent(userText);
    if (localIntent.matched) {
      return res.json({ blocked: false, content: localIntent.reply, triage: null, safetyStage: "local-intent", sessionId: activeSessionId });
    }

    // Step 2: Precheck
    const precheck = preCheckUserInput(userText);
    if (precheck.blocked) {
      return res.json({ blocked: true, content: precheck.safeReply, triage: "self-care", safetyStage: "precheck", sessionId: activeSessionId });
    }

    // Add user message to session
    session.messages.push({ role: "user", content: userText });

    // Step 3: Pre-risk classifier
    const preRisk = await runPreRiskClassifier(session.messages, HF_TOKEN);
    if (preRisk.prohibited) {
      return res.json({
        blocked: true,
        content: "I'm not allowed to provide diagnoses, prescriptions, or specific treatments.\nI can give general information about eye symptoms and when to see an eye doctor.",
        triage: "routine-ophthalmologist",
        safetyStage: "pre-risk-prohibited",
        sessionId: activeSessionId,
      });
    }

    // Risk context overlay
    const riskSystemMessage = {
      role: "system",
      content: `[SAFETY_RISK_STATE]\nrisk_level: ${preRisk.risk_level}\nrequest_type: ${preRisk.request_type}\nrequires_clarification: ${preRisk.requires_clarification}\n\nYou are an ophthalmology assistant. Avoid diagnosis or medication. Ask clarifying questions first if requires_clarification is true. If risk_level is high or emergency, strongly recommend urgent eye care.`,
    };

    const assistantMessages = [riskSystemMessage, ...session.messages];

    // Step 4: Call Azure assistant
    const rawAssistantText = await callAzureAssistant(assistantMessages, mode, azureConfig);

    // Step 5: Extract TRIAGE_JSON and clean text
    const triageInfo = parseTriageJsonFromText(rawAssistantText);
    const cleanAnswer = removeTriageJsonBlock(rawAssistantText);

    session.messages.push({ role: "assistant", content: cleanAnswer });

    // Step 6: Clarifying vs final answer detection
    const isAssistantQuestion = cleanAnswer.trim().endsWith("?");
    const assistantClarifyingPatterns = ["one last question", "to better understand", "to help understand", "could you tell me", "may i ask", "please clarify"];
    const looksLikeClarifying = isAssistantQuestion || assistantClarifyingPatterns.some((p) => cleanAnswer.toLowerCase().includes(p));

    const shortAnswerKeywords = ["yes", "no", "both", "one", "left", "right", "maybe", "not sure", "none"];
    const isShortUserAnswer = lowerUser.split(/\s+/).length <= 3 && shortAnswerKeywords.some((w) => lowerUser.startsWith(w));

    if (triageInfo.ok && looksLikeClarifying) {
      return res.json({ blocked: false, content: cleanAnswer, triage: null, safetyStage: "clarifying-overrode-json", sessionId: activeSessionId });
    }

    if (!triageInfo.ok && (looksLikeClarifying || isShortUserAnswer)) {
      return res.json({ blocked: false, content: cleanAnswer, triage: null, safetyStage: looksLikeClarifying ? "clarifying-question" : "clarifying-user-answer", sessionId: activeSessionId });
    }

    if (!triageInfo.ok && preRisk.requires_clarification) {
      return res.json({ blocked: false, content: cleanAnswer, triage: null, safetyStage: "clarifying-preRisk", sessionId: activeSessionId });
    }

    // Step 7: Final stage – TRIAGE_JSON is preferred but not required
    // If missing, return the assistant's actual response with fallback triage
    if (!triageInfo.ok) {
      console.log("[TRIAGE] Missing TRIAGE_JSON in response. Using fallback triage and returning actual response.");

      // Risk-aware triage for missing JSON block
      const missingJsonFallbackTriage = getFallbackTriage(preRisk);

      // Check for safety issues in the response even without TRIAGE_JSON
      if (containsMedicationLanguage(cleanAnswer)) {
        console.log("[FALLBACK] Medication language detected in response without TRIAGE_JSON.");
        return res.json({
          blocked: false,
          content:
            "I can give general information, but I cannot provide treatments, medications, or dosages.\nPlease consult an ophthalmologist.",
          triage: missingJsonFallbackTriage,
          safetyStage: "fallback",
          sessionId: activeSessionId,
        });
      }

      if (looksLikeExamHallucination(cleanAnswer)) {
        console.log("[FALLBACK] Hallucination detected in response without TRIAGE_JSON.");
        return res.json({
          blocked: false,
          content:
            "I cannot examine your eyes or see any images or tests. I can only provide general information based on what you describe.\nFor an accurate assessment, please visit an ophthalmologist.",
          triage: missingJsonFallbackTriage,
          safetyStage: "hallucination-blocked",
          sessionId: activeSessionId,
        });
      }

      // Store assistant answer in session for conversation continuity
      session.messages.push({ role: "assistant", content: cleanAnswer });

      // Return the actual assistant response with fallback triage
      return res.json({
        blocked: false,
        content: cleanAnswer,
        triage: missingJsonFallbackTriage,
        safetyStage: "missing-json-fallback",
        sessionId: activeSessionId,
      });
    }

    // Step 8: Post-answer safety classifier
    const postSafety = await runPostSafetyClassifier(userText, cleanAnswer, triageInfo, HF_TOKEN);

    if (postSafety.needs_block) {
      return res.json({
        blocked: false,
        content: "I'm not able to answer that safely. I can only provide general information about eye symptoms.\nPlease consult an ophthalmologist for a proper examination.",
        triage: "routine-ophthalmologist",
        safetyStage: "post-safety-blocked",
        sessionId: activeSessionId,
      });
    }

    if (postSafety.safety === "hallucination_exam" || looksLikeExamHallucination(cleanAnswer)) {
      return res.json({
        blocked: false,
        content: "I cannot examine your eyes or see any images or tests. I can only provide general information based on what you describe.\nFor an accurate assessment, please visit an ophthalmologist.",
        triage: "routine-ophthalmologist",
        safetyStage: "hallucination-blocked",
        sessionId: activeSessionId,
      });
    }

    // Step 9: Medication fallback
    if (containsMedicationLanguage(cleanAnswer) || triageInfo.mentions_meds) {
      return res.json({
        blocked: false,
        content: "I can give general information, but I cannot provide treatments, medications, or dosages.\nPlease consult an ophthalmologist.",
        triage: "routine-ophthalmologist",
        safetyStage: "fallback",
        sessionId: activeSessionId,
      });
    }

    // Step 10: Successful final output
    const correctedTriage = validateAndCorrectTriage(triageInfo, preRisk, postSafety);

    return res.json({
      blocked: false,
      content: cleanAnswer,
      triage: correctedTriage,
      red_flags_detected: triageInfo.red_flags_detected,
      mentions_meds: triageInfo.mentions_meds,
      safetyStage: "ok",
      risk_level: preRisk.risk_level,
      sessionId: activeSessionId,
    });
  } catch (err) {
    console.error("[SERVER] Unhandled error:", err);
    return res.json({
      blocked: true,
      content: "Something went wrong. If your symptoms are new, severe, or getting worse, please seek urgent eye care.",
      triage: "urgent-24h",
      safetyStage: "exception",
    });
  }
});

// Health Check
app.get("/api/health", (req, res) => {
  res.json({ ok: true, message: "Eye Symptom Assistant API is running on Firebase" });
});

// Export the Express app as a Firebase Cloud Function
export const api = onRequest(
  {
    secrets: [azureEndpoint, azureApiKey, azureDeployment, hfToken],
    memory: "512MiB",
    timeoutSeconds: 120,
    region: "europe-west1",
  },
  app
);

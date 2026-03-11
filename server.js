// Eye Symptom Assistant – Backend Server

import express from "express";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import cors from "cors";
import crypto from "crypto";

dotenv.config();

// Express Initialization

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.static(path.join(__dirname, "public")));
app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

//  Configuration

/** Azure OpenAI configuration (assistant model) */
const AZURE_ENDPOINT = process.env.AZURE_OPENAI_ENDPOINT?.replace(/\/+$/, "");
const AZURE_API_KEY = process.env.AZURE_OPENAI_API_KEY;
const AZURE_DEPLOYMENT = process.env.AZURE_OPENAI_DEPLOYMENT;
const AZURE_API_VERSION =
  process.env.AZURE_API_VERSION || "2024-02-15-preview";

/** Hugging Face token for Llama classifiers */
const HF_TOKEN = process.env.HF_TOKEN;

/** Llama model used for both pre-risk and post-safety classifiers */
const LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct";

/** HTTP server port */
const PORT = process.env.PORT || 3000;

//  External API Wrappers

function buildAzureUrl() {
  return (
    `${AZURE_ENDPOINT}/openai/deployments/${AZURE_DEPLOYMENT}` +
    `/chat/completions?api-version=${AZURE_API_VERSION}`
  );
}

/**
 * Calls Azure Chat Completions for the main assistant.
 *
 * @param {Array} messages - Chat messages (system, user, assistant)
 * @param {string} mode - Optional mode ("research" or default)
 * @returns {Promise<string>} assistant content
 */
async function callAzureAssistant(messages, mode) {
  const url = buildAzureUrl();
  const body = {
    messages,
    temperature: mode === "research" ? 0.5 : 0.2,
    max_tokens: 700,
  };

  console.log("[AZURE] Requesting assistant completion...");

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "api-key": AZURE_API_KEY,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error("[AZURE] Error:", res.status, err);
    throw new Error("Azure assistant call failed");
  }

  const json = await res.json();
  const content = json.choices?.[0]?.message?.content || "";
  console.log("[AZURE] Assistant response received.");
  return content;
}

/**
 * Calls Hugging Face Chat Completions with Llama (JSON-only classifiers).
 *
 * @param {Array} messages - Chat messages for classifier
 * @param {number} maxTokens - Maximum tokens for the classifier output
 * @returns {Promise<Object>} parsed JSON from the classifier
 */
async function callLlamaClassifier(messages, maxTokens = 256) {
  if (!HF_TOKEN) {
    throw new Error("HF_TOKEN missing for Llama classifiers");
  }

  console.log("[LLAMA] Requesting classifier output...");

  const res = await fetch(
    "https://router.huggingface.co/v1/chat/completions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: LLAMA_MODEL,
        temperature: 0,
        max_tokens: maxTokens,
        messages,
      }),
    }
  );

  if (!res.ok) {
    const err = await res.text();
    console.error("[LLAMA] Error:", res.status, err);
    throw new Error("Llama classifier call failed");
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || "";

  try {
    const parsed = JSON.parse(content);
    console.log("[LLAMA] JSON parsed successfully.");
    return parsed;
  } catch (e) {
    console.warn("[LLAMA] JSON parse failed. Raw content:", content);
    throw new Error("Llama JSON parse failed");
  }
}

// Local Helpers, System Prompt, and Safety Utilities

/**
 * Main assistant system prompt - UPDATED with q_asked constraint
 */
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
- Ask a maximum of 3 clarifying questions in total. **(New Constraint)**
- If the number of questions asked is less than 3, ask ONE clarifying question.
- If the number of questions asked is 3 or more, OR if enough detail is already present/user asks to skip questions, you MUST provide the final response including the TRIAGE_JSON block.
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
1. Never diagnose or say “you have”.
2. Never recommend medications, eye drops, dosages, treatments.
3. Never name medications.
4. Use phrases like “common possibilities include…”.
5. Always end final response with:
   “This is general information only — not a diagnosis.
    See an ophthalmologist if symptoms persist or worry you.”

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

/**
 * In-memory session store: sessionId -> { messages: [...], q_asked: 0 }
 * messages always start with the system prompt.
 */
const sessions = new Map();

function getOrCreateSession(sessionId) {
  let id = sessionId;
  if (!id || !sessions.has(id)) {
    id = id || crypto.randomUUID();
    sessions.set(id, {
      messages: [
        {
          role: "system",
          content: ASSISTANT_SYSTEM_PROMPT,
        },
      ],
      // New state tracker for explicit question control
      q_asked: 0, 
    });
  }
  return { id, session: sessions.get(id) };
}

/**
 * Basic local precheck:
 * - empty input
 * - medication / dosage
 * - self-harm expressions
 */
function preCheckUserInput(text) {
  if (!text || typeof text !== "string") {
    console.log("[PRECHECK] Empty user input.");
    return {
      blocked: true,
      safeReply:
        "I couldn't read your message. Please describe your eye concern.",
    };
  }

  const lower = text.toLowerCase();

  // Medication-related patterns (blocked at precheck)
  const medicationPatterns = [
    " mg ",
    "mg/",
    "dosage",
    "dose",
    "prescribe",
    "prescription",
    "antibiotic",
    "steroid",
    "eye drops with",
    "drops of",
    "tablet",
    "pill",
  ];

  if (medicationPatterns.some((p) => lower.includes(p))) {
    console.log("[PRECHECK] Medication or dosage request blocked.");
    return {
      blocked: true,
      safeReply:
        "I’m not able to discuss medications or dosages. I can only give general information about eye symptoms.",
    };
  }

  // Very simple self-harm screen
  const selfHarmPatterns = [
    "kill myself",
    "suicide",
    "end my life",
    "hurt myself",
    "don't want to live",
  ];

  if (selfHarmPatterns.some((p) => lower.includes(p))) {
    console.log("[PRECHECK] Self-harm language detected.");
    return {
      blocked: true,
      safeReply:
        "I'm really sorry that you're feeling this way. I’m not able to help with self-harm or suicide situations.\n" +
        "Please contact your local emergency number or a mental health helpline immediately.",
    };
  }

  return { blocked: false };
}

/**
 * Simple local greeting / closing detection so we do not hit the LLM for trivial turns.
 */
function detectLocalIntent(text) {
  const lower = text.trim().toLowerCase();

  const greetings = ["hi", "hello", "hey", "good morning", "good evening"];
  if (greetings.some((g) => lower.startsWith(g))) {
    return {
      matched: true,
      reply:
        "Hello! I'm your Eye Health Assistant.\n" +
        "I can help with visual acuity tests and questions about eye symptoms.\n" +
        "How can I help your eyes today?",
    };
  }

  const closings = ["ok", "okay", "thanks", "thank you", "bye", "great", "good"];
  if (closings.some((c) => lower.startsWith(c))) {
    return {
      matched: true,
      reply:
        "You're welcome! If anything else bothers your eyes, feel free to ask!",
    };
  }

  return { matched: false };
}

/**
 * Serializes the conversation for the classifier (simple text format).
 */
function serializeConversation(messages) {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n");
}

/**
 * Extracts TRIAGE_JSON from the assistant response if present.
 */
function parseTriageJsonFromText(text) {
  const result = {
    ok: false,
    triage: "unknown",
    red_flags_detected: [],
    mentions_meds: false,
  };

  const match = text.match(/TRIAGE_JSON:\s*({[\s\S]*})/);

  if (!match) {
    console.log("[TRIAGE] No TRIAGE_JSON found.");
    return result;
  }

  try {
    const parsed = JSON.parse(match[1]);
    console.log("[TRIAGE] TRIAGE_JSON parsed successfully.");
    return {
      ok: true,
      triage: parsed.triage ?? "unknown",
      red_flags_detected: parsed.red_flags_detected ?? [],
      mentions_meds: parsed.mentions_meds ?? false,
    };
  } catch {
    console.log("[TRIAGE] Failed to parse TRIAGE_JSON.");
    return result;
  }
}

/**
 * Removes TRIAGE_JSON block from the assistant response text.
 */
function removeTriageJsonBlock(text) {
  return text.replace(/TRIAGE_JSON:[\s\S]*$/, "").trim();
}

/**
 * Simple medication keyword filter on final answers.
 */
function containsMedicationLanguage(text) {
  const lower = text.toLowerCase();
  const patterns = [
    " mg ",
    "mg/",
    "dosage",
    "antibiotic",
    "steroid",
    "take ",
    "tablet",
    "pill",
  ];
  return patterns.some((p) => lower.includes(p));
}

/**
 * Rule-based hallucination detector for fake exams / scans.
 */
function looksLikeExamHallucination(text) {
  const lower = text.toLowerCase();
  const patterns = [
    "after examining your eye",
    "on your eye exam",
    "i have examined your eye",
    "i have seen your retina",
    "based on your scan",
    "your blood test shows",
    "i checked your test results",
  ];
  return patterns.some((p) => lower.includes(p));
}

// Classifiers (Pre-Risk and Post-Safety, using Llama)

async function runPreRiskClassifier(messages) {
  try {
    if (!HF_TOKEN) {
      throw new Error("HF_TOKEN missing");
    }

    const convo = serializeConversation(messages);
    const classifierMessages = [
      {
        role: "system",
content:
  "You are a medical RISK classifier for an ophthalmology assistant.\n" +
  "Read the conversation and output ONLY valid JSON with exactly these fields:\n" +
  "{\n" +
  '  "risk_level": "low" | "moderate" | "high" | "emergency",\n' +
  '  "request_type": "information" | "diagnosis_request" | "treatment_request" | "triage" | "self-harm" | "other",\n' +
  '  "requires_clarification": true | false,\n' +
  '  "prohibited": true | false,\n' +
  '  "notes": string\n' +
  "}\n\n" +

  "IMPORTANT INTENT RULES:\n" +
  "- Describing symptoms (e.g. 'my eyes are red and swollen') is NOT a diagnosis request.\n" +
  "- Asking 'what do I have?', naming a disease, or requesting confirmation IS a diagnosis request.\n" +
  "- prohibited MUST be true ONLY if the user explicitly asks for:\n" +
  "  * a diagnosis\n" +
  "  * medications or dosages\n" +
  "  * prescriptions or treatments\n" +
  "  * self-harm instructions\n" +
  "- If the user only reports symptoms, prohibited MUST be false.\n" +
  "- If symptom information is incomplete, set requires_clarification = true.\n\n" +

  "Output ONLY valid JSON. Do not include explanations."
,
      },
      {
        role: "user",
        content: `CONVERSATION:\n${convo}`,
      },
    ];

    const json = await callLlamaClassifier(classifierMessages, 220);

    const result = {
      risk_level: json.risk_level || "low",
      request_type: json.request_type || "other",
      requires_clarification:
        typeof json.requires_clarification === "boolean"
          ? json.requires_clarification
          : false,
      prohibited: !!json.prohibited,
      notes: json.notes || "",
    };

    console.log("[PRE-RISK] Result:", result);
    return result;
  } catch (err) {
    console.warn("[PRE-RISK] Failed, using safe defaults:", err.message);
    return {
      risk_level: "low",
      request_type: "other",
      requires_clarification: false,
      prohibited: false,
      notes: "",
    };
  }
}

async function runPostSafetyClassifier(userText, answerText, triageInfo) {
  try {
    if (!HF_TOKEN) {
      throw new Error("HF_TOKEN missing");
    }

    const triageJson = JSON.stringify(triageInfo || {});
    const classifierMessages = [
      {
        role: "system",
        content:
  "You are a medical SAFETY classifier for an ophthalmology assistant.\n" +
  "Given a user symptom description, an assistant reply, and triage JSON, output ONLY valid JSON:\n" +
  "{\n" +
  '  "safety": "ok" | "undertriage" | "overtriage" | "medication" | "diagnosis" | "hallucination_exam" | "other",\n' +
  '  "needs_block": true | false,\n' +
  '  "needs_upgrade_to_emergency": true | false,\n' +
  '  "reasons": [string]\n' +
  "}\n" +
  "IMPORTANT RULES:\n" +
  "- General lifestyle advice is ALLOWED and SAFE.\n" +
  "- Examples of allowed advice:\n" +
  "  * screen breaks (e.g. 20-20-20 rule)\n" +
  "  * blinking reminders\n" +
  "  * rest, posture, lighting adjustments\n" +
  "- These MUST NOT be classified as medication or treatment.\n" +
  "- Medication means: drugs, eye drops, ointments, dosages, prescriptions.\n" +
  "- undertriage applies ONLY if serious red-flag symptoms are present and urgency is minimized.\n" +
  "- hallucination_exam applies ONLY if the assistant claims to have examined the patient.\n" +
  "Do NOT output anything besides this JSON."
,
      },
      {
        role: "user",
        content:
          `USER_SYMPTOMS:\n${userText}\n\n` +
          `ASSISTANT_REPLY:\n${answerText}\n\n` +
          `TRIAGE_JSON:\n${triageJson}`,
      },
    ];

    const json = await callLlamaClassifier(classifierMessages, 220);

    const result = {
      safety: json.safety || "ok",
      needs_block: !!json.needs_block,
      needs_upgrade_to_emergency: !!json.needs_upgrade_to_emergency,
      reasons: Array.isArray(json.reasons) ? json.reasons : [],
    };

    console.log("[POST-SAFETY] Result:", result);
    return result;
  } catch (err) {
    console.warn("[POST-SAFETY] Failed, defaulting to ok:", err.message);
    return {
      safety: "ok",
      needs_block: false,
      needs_upgrade_to_emergency: false,
      reasons: [],
    };
  }
}

//  Triage Correction / Validation

function validateAndCorrectTriage(triageInfo, preRisk, answerSafety) {
  let triage = triageInfo.triage || "unknown";

  if (triageInfo.red_flags_detected.length > 0 && triage === "self-care") {
    console.log("[TRIAGE] Upgrading triage due to red flags.");
    triage = "urgent-24h";
  }

  if (
    preRisk &&
    (preRisk.risk_level === "high" || preRisk.risk_level === "emergency")
  ) {
    if (triage === "self-care" || triage === "routine-ophthalmologist") {
      console.log("[TRIAGE] Upgrading triage due to pre-risk high/emergency.");
      triage =
        preRisk.risk_level === "emergency" ? "emergency-now" : "urgent-24h";
    }
  }

  if (answerSafety && answerSafety.needs_upgrade_to_emergency) {
    console.log("[TRIAGE] Upgrading triage due to post-safety escalation.");
    triage = "emergency-now";
  }

  return triage;
}

/**
 * Helper to determine the safest fallback triage based on preRisk.
 * @param {Object} preRisk - Result from runPreRiskClassifier
 * @returns {string} The appropriate triage level.
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


// Main Chat Endpoint  (session-based)

app.post("/api/chat", async (req, res) => {
  try {
    let { sessionId, message, mode } = req.body;
    mode = mode || "normal";

    const userText = (message || "").toString().trim();
    const lowerUser = userText.toLowerCase();

    // Create or load session
    const { id: activeSessionId, session } = getOrCreateSession(sessionId);

    // 1: Local greeting/closing handling (no LLM call)
    const localIntent = detectLocalIntent(userText);
    if (localIntent.matched) {
      console.log("[LOCAL] Greeting or closing detected.");
      return res.json({
        blocked: false,
        content: localIntent.reply,
        triage: null,
        safetyStage: "local-intent",
        sessionId: activeSessionId,
      });
    }

    // 2: Precheck (empty, medication, self-harm)
    const precheck = preCheckUserInput(userText);
    if (precheck.blocked) {
      return res.json({
        blocked: true,
        content: precheck.safeReply,
        triage: "self-care",
        safetyStage: "precheck",
        sessionId: activeSessionId,
      });
    }

    // Add user message to session
    session.messages.push({ role: "user", content: userText });

    // 3: Pre-risk classifier (Llama)
    const preRisk = await runPreRiskClassifier(session.messages);
    
    // Risk-aware triage for prohibited requests
    const prohibitedFallbackTriage = getFallbackTriage(preRisk);
    
    if (preRisk.prohibited) {
      console.log("[PRE-RISK] Prohibited request (diagnosis/treatment/self-harm).");
      return res.json({
        blocked: true,
        content:
          "I’m not allowed to provide diagnoses, prescriptions, or specific treatments.\n" +
          "I can give general information about eye symptoms and when to see an eye doctor.",
        triage: prohibitedFallbackTriage,
        safetyStage: "pre-risk-prohibited",
        sessionId: activeSessionId,
      });
    }

    // Risk context “overlay” for assistant, includes the q_asked count
    const riskSystemMessage = {
      role: "system",
      content:
        "[SAFETY_RISK_STATE]\n" +
        `q_asked_count: ${session.q_asked}\n` + // Pass the counter to the main LLM
        `risk_level: ${preRisk.risk_level}\n` +
        `request_type: ${preRisk.request_type}\n` +
        `requires_clarification: ${preRisk.requires_clarification}\n\n` +
        "You are an ophthalmology assistant. You must:\n" +
        "- Avoid diagnosis or medication.\n" +
        "- Ask clarifying questions first if requires_clarification is true AND q_asked_count is less than 3.\n" +
        "- If risk_level is high or emergency, strongly recommend urgent eye care, never pure self-care.\n" +
        "- Keep answers short, clear, and empathetic.",
    };

    const assistantMessages = [riskSystemMessage, ...session.messages];

    // 4: Call Azure assistant
    const rawAssistantText = await callAzureAssistant(assistantMessages, mode);

    // 5: Extract TRIAGE_JSON and clean text
    const triageInfo = parseTriageJsonFromText(rawAssistantText);
    const cleanAnswer = removeTriageJsonBlock(rawAssistantText);

    // 6: Clarifying vs final answer detection
    
    // Check if the output is a clarifying question (heuristic, as a fallback)
    const isAssistantQuestion = cleanAnswer.trim().endsWith("?");
    const looksLikeClarifying = isAssistantQuestion || cleanAnswer.length < 150 && !triageInfo.ok;


    // Logic for Clarifying Stage: Triage JSON is not present AND 
    // we have explicitly asked fewer than 3 questions OR 
    // the pre-risk classifier still requests more clarification.
    if (!triageInfo.ok && (session.q_asked < 3 || preRisk.requires_clarification)) {
      
      // If the model output a question, increment the counter
      if (looksLikeClarifying) {
          session.q_asked += 1;
          console.log(`[CLARIFY] Clarifying question detected. New count: ${session.q_asked}`);
      } else {
          // If the model failed to output a question but didn't output JSON, treat it as part of the clarifying phase
          console.log("[CLARIFY] Assistant response is text but missing JSON/Question. Continuing clarification phase.");
      }
      
      // Store assistant answer (without TRIAGE_JSON) back into session
      session.messages.push({ role: "assistant", content: cleanAnswer });

      return res.json({
        blocked: false,
        content: cleanAnswer,
        triage: null,
        safetyStage: "clarifying-in-progress",
        sessionId: activeSessionId,
      });
    }

    // Explicitly set clarification complete if we reach here and TRIAGE_JSON is required
    session.q_asked = 3; 

    // 7: Final stage – TRIAGE_JSON is preferred but not required
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

    // Store assistant answer (without TRIAGE_JSON) back into session
    session.messages.push({ role: "assistant", content: cleanAnswer });

    // 8: Post-answer safety classifier
    const postSafety = await runPostSafetyClassifier(
      userText,
      cleanAnswer,
      triageInfo
    );

    // Risk-aware triage for post-safety block
    const postSafetyFallbackTriage = getFallbackTriage(preRisk);
    
    if (postSafety.needs_block) {
      console.log("[POST-SAFETY] Final answer blocked by safety classifier.");
      return res.json({
        blocked: false,
        content:
          "I’m not able to answer that safely. I can only provide general information about eye symptoms.\n" +
          "Please consult an ophthalmologist for a proper examination.",
        triage: postSafetyFallbackTriage,
        safetyStage: "post-safety-blocked",
        sessionId: activeSessionId,
      });
    }

    // Extra hallucination net
    if (
      postSafety.safety === "hallucination_exam" ||
      looksLikeExamHallucination(cleanAnswer)
    ) {
      console.log("[HALLUCINATION] Exam or test hallucination detected.");
      // Risk-aware triage for hallucination block
      const hallucinationFallbackTriage = getFallbackTriage(preRisk);

      return res.json({
        blocked: false,
        content:
          "I cannot examine your eyes or see any images or tests. I can only provide general information based on what you describe.\n" +
          "For an accurate assessment, please visit an ophthalmologist.",
        triage: hallucinationFallbackTriage,
        safetyStage: "hallucination-blocked",
        sessionId: activeSessionId,
      });
    }

    // 9: Medication fallback
    if (containsMedicationLanguage(cleanAnswer) || triageInfo.mentions_meds) {
      console.log("[FALLBACK] Medication or dosage language detected in final answer.");
      // Risk-aware triage for medication fallback
      const medicationFallbackTriage = getFallbackTriage(preRisk);

      return res.json({
        blocked: false,
        content:
          "I can give general information, but I cannot provide treatments, medications, or dosages.\nPlease consult an ophthalmologist.",
        triage: medicationFallbackTriage,
        safetyStage: "fallback",
        sessionId: activeSessionId,
      });
    }

    // 10: Successful final output with triage correction
    const correctedTriage = validateAndCorrectTriage(
      triageInfo,
      preRisk,
      postSafety
    );

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
      content:
        "Something went wrong. If your symptoms are new, severe, or getting worse, please seek urgent eye care.",
      triage: "urgent-24h",
      safetyStage: "exception",
    });
  }
});

//  Health Check Endpoint

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    azureConfigured: !!AZURE_API_KEY,
    hfConfigured: !!HF_TOKEN,
  });
});

//  Start Server

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Azure deployment: ${AZURE_DEPLOYMENT}`);
  console.log(
    `Llama classifiers: ${HF_TOKEN ? "ENABLED (Llama 3.1 8B)" : "DISABLED"}`
  );
});
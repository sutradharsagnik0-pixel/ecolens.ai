import express from "express";
import path from "path";
import { GoogleGenAI, Type, ThinkingLevel } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const PORT = 3000;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is missing.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

/**
 * Execute Gemini model generation with resilient fallback and low-latency thinking.
 * First tries gemini-3.8-flash with ThinkingLevel.LOW (fastest reasoning without delay);
 * if unavailable, seamlessly falls back to gemini-3.1-flash-lite.
 */
async function generateContentWithFallback(
  ai: GoogleGenAI,
  params: any
) {
  const candidateModels = ["gemini-3.8-flash", "gemini-3.1-flash-lite"];
  let lastError: any = null;

  for (const model of candidateModels) {
    try {
      const config = { ...(params.config || {}) };
      // Apply ThinkingLevel.LOW to gemini-3.8-flash to minimize latency while maintaining high precision
      if (model === "gemini-3.8-flash") {
        config.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      } else {
        delete config.thinkingConfig;
      }

      const res = await ai.models.generateContent({
        ...params,
        config,
        model,
      });
      if (res && res.text) {
        return res;
      }
    } catch (err: any) {
      console.warn(`[EcoLens Server] Model ${model} encountered issue (${err?.status || err?.message}). Trying next candidate...`);
      lastError = err;
      // Brief pause before fallback candidate
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }

  throw lastError || new Error("All AI models currently unavailable. Please retry in a moment.");
}

function sanitizeBase64(imageBase64: string): string {
  let cleaned = (imageBase64 || "").trim();
  const commaIdx = cleaned.indexOf(",");
  if (cleaned.startsWith("data:") && commaIdx !== -1) {
    cleaned = cleaned.slice(commaIdx + 1);
  }
  return cleaned.replace(/\s+/g, "");
}

function normalizeMimeType(mimeType?: string): string {
  let mime = (mimeType || "image/jpeg").toLowerCase().trim();
  if (mime === "image/jpg") mime = "image/jpeg";
  if (!["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"].includes(mime)) {
    mime = "image/jpeg";
  }
  return mime;
}

function extractJson(text: string): any {
  let clean = (text || "").trim();
  if (clean.startsWith("```")) {
    clean = clean.replace(/^```[a-zA-Z]*\n?/, "").replace(/\n?```$/, "").trim();
  }
  return JSON.parse(clean);
}

async function startServer() {
  const app = express();

  // Increase payload size for base64 camera images
  app.use(express.json({ limit: "35mb" }));
  app.use(express.urlencoded({ extended: true, limit: "35mb" }));

  // Health check
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok", service: "EcoLens AI Server" });
  });

  // AI Analysis Endpoint
  app.post("/api/analyze", async (req, res) => {
    try {
      const {
        imageBase64,
        mimeType = "image/jpeg",
        language = "en",
        isVideo = false,
        frames = [],
        videoBase64 = null,
      } = req.body;

      let primaryImageBase64 = imageBase64;
      if (!primaryImageBase64 && Array.isArray(frames) && frames.length > 0) {
        primaryImageBase64 = frames[0].base64 || frames[0].data;
      }

      if (!primaryImageBase64) {
        return res.status(400).json({ error: "No image or video frame payload provided" });
      }

      // Clean and sanitize base64 payload
      const cleanedBase64 = sanitizeBase64(primaryImageBase64);
      if (!cleanedBase64) {
        return res.status(400).json({ error: "Invalid or empty image data provided" });
      }
      const resolvedMime = normalizeMimeType(mimeType);

      const ai = getGeminiClient();

      const langInstructionsMap: Record<string, string> = {
        en: "Respond entirely in English with clear, helpful, educational language.",
        hi: "सभी विवरण, घटक विश्लेषण, निर्देश और वॉयस स्क्रिप्ट स्वाभाविक हिंदी (Hindi in Devanagari script) में लिखें।",
        bn: "সকল বিবরণ, উপাদান বিশ্লেষণ, পদক্ষেপ ও ভয়েস স্ক্রিপ্ট সাবলীল বাংলায় (Bengali in Bengali script) লিখুন।",
        mr: "सर्व तपशील, घटक विश्लेषण, कृती योजना आणि व्हॉइस स्क्रिप्ट अस्खलित मराठीत (Marathi in Devanagari script) लिहा.",
        te: "అన్ని వివరాలు, విశ్లేషణ, సూచనలు మరియు వాయిస్ స్క్రిప్ట్ స్వచ్ఛమైన తెలుగులో (Telugu script) వ్రాయండి.",
        ta: "அனைத்து விவரங்கள், பகுப்பாய்வு, வழிமுறைகள் மற்றும் குரல் ஸ்கிரிப்டை தூய தமிழில் (Tamil script) எழுதவும்.",
        gu: "તમામ વિગતો, ઘટક વિશ્લેષણ, સૂચનાઓ અને વૉઇસ સ્ક્રિપ્ટ અસ્ખલિત ગુજરાતીમાં (Gujarati script) લખો.",
        ur: "تمام تفصیلات، تجزیہ، تجاویز اور صوتی اسکرپٹ روانی سے اردو (Urdu script) میں لکھیں۔",
        kn: "ಎಲ್ಲಾ ವಿವರಗಳು, ವಿಶ್ಲೇಷಣೆ, ಸಲಹೆಗಳು ಮತ್ತು ಧ್ವನಿ ಸ್ಕ್ರಿಪ್ಟ್ ಅನ್ನು ಸ್ಪಷ್ಟ ಕನ್ನಡದಲ್ಲಿ (Kannada script) ಬರೆಯಿರಿ.",
        or: "ସମସ୍ତ ବିବରଣୀ, ବିଶ୍ଳେଷଣ ଏବଂ ଭଏସ୍ ସ୍କ୍ରିପ୍ଟ ସ୍ପଷ୍ଟ ଓଡ଼ିଆରେ (Odia script) ଲେଖନ୍ତୁ।",
        ml: "എല്ലാ വിവരങ്ങളും ഘടക വിശകലനങ്ങളും വോയ്‌സ് സ്ക്രിപ്റ്റും വ്യക്തമായ മലയാളത്തിൽ (Malayalam script) എഴുതുക.",
        pa: "ਸਾਰੇ ਵੇਰਵੇ, ਵਿਸ਼ਲੇਸ਼ਣ, ਹਦਾਇਤਾਂ ਅਤੇ ਵਾਇਸ ਸਕ੍ਰਿਪਟ ਸਪਸ਼ਟ ਪੰਜਾਬੀ (Gurmukhi script) ਵਿੱਚ ਲਿਖੋ।",
        as: "সকলো বিৱৰণ, উপাদান বিশ্লেষণ আৰু ভইচ স্ক্রিপ্ট স্পষ্ট অসমীয়াত (Assamese script) লিখক।",
        mai: "सभटा विवरण, घटक विश्लेषण, निर्देश आ वॉयस स्क्रिप्ट मैथिली (Devanagari script) मे लिखू।",
        sa: "सर्वं विवरणम्, घटक-विश्लेषणं, निर्देशान्, ध्वनि-स्क्रिप्टं च संस्कृतेन (Devanagari script) लिखतु।",
      };
      const langInstructions = langInstructionsMap[language] || "Respond entirely in English.";

      const systemPrompt = `You are EcoLens AI, an ultra-intelligent, omni-domain computer vision, sustainability, and circular economy valuation intelligence system.
Your mission is "See It. Understand It. Reuse It. Recycle It."

CRITICAL DIRECTIVES:

1. UNIVERSAL WORLD OBJECT RECOGNITION (SCAN ANYTHING EXISTING IN THE WORLD):
   - You MUST analyze and identify ANY physical item or material existing in the world that a user photographs:
     * Consumer electronics & e-waste (laptops, phones, mice, keyboards, chargers, cables, circuit boards, batteries, appliances, gadgets, switches)
     * Domestic packaging & containers (plastic dispensers, toothpick containers, pill boxes, bottles, caps, blister packs, cartons, jars, cans)
     * Scrap metals & industrial parts (copper wire, copper pipes, brass fittings, aluminum cans/profiles, cast iron, steel rebar, lead plates, stainless steel)
     * Plastics of all grades (PET #1, HDPE #2, PVC #3, LDPE #4, PP #5, PS #6, Other #7 including ABS, Polycarbonate, Acrylic)
     * Paper & Cardboard (corrugated boxes, kraft paper, duplex cartons, office paper, books, newspapers, tetra paks)
     * Glass & Ceramics (bottles, jars, cullet, porcelain, ceramic tableware, tiles)
     * Protective headgear & sports gear (motorcycle helmets, bicycle helmets, hard hats, padding)
     * Textiles, Rubber & Leather (garments, shoes, belts, tires, rubber hoses, fabric scraps)
     * Wood, Bamboo & Organics (furniture, pallets, chopsticks, toothpicks, organic biomass)
     * Hardware, Tools & Automotive (wrenches, pliers, bolts, automotive components)
   - NEVER reject or fail an item because it is not e-waste. EcoLens AI is the universal waste, recycling, and scrap intelligence for the entire physical world.

STRICT ANTI-HALLUCINATION RULES FOR ELECTRONICS & BATTERIES:
   - YOU MUST ONLY identify components that ACTUALLY physically exist in the photographed item.
   - ABSOLUTELY NEVER attribute electronic components, PCBs, circuits, lithium batteries, cells, wiring, or transformers to non-electronic everyday items!
   - Specific example: MOTORCYCLE HELMET / BICYCLE HELMET / SAFETY HEADGEAR:
     * 1. Outer Impact Shell (Polycarbonate PC / ABS / Fiberglass / Carbon Fiber composite)
     * 2. Shock Absorber Liner (Expanded Polystyrene EPS Foam / Densified Thermocol)
     * 3. Comfort Cheek Pads & Interior Fabric (Polyurethane foam + Polyester textile liner)
     * 4. Retention Chinstrap & Quick-Release/D-Ring Buckle (Nylon webbing + Stainless Steel / Aluminum D-rings)
     * 5. Optical Face Shield / Visor (Polycarbonate with anti-scratch coating)
     * 6. Visor Pivot Assembly & Screws (ABS base plates + Steel pivot screws)
     * 7. Ventilation Ducts & Exhaust Louvers (Polypropylene / ABS molded trim)
     * A standard helmet DOES NOT CONTAIN ANY LITHIUM BATTERY, CIRCUIT BOARD, MOTOR, OR CAPACITOR!
     * NEVER output battery fire warnings, battery handling notes, or e-waste recycling for helmets.
     * Environmental hazards for helmets MUST be EPS foam and rigid plastic landfill persistence (taking 500+ years to degrade) or toxic styrene smoke if burned—NOT battery fire or acid leakage.
     * Recyclers and action plan for helmets MUST be dedicated plastic granulators, EPS densifiers, or helmet retirement centers—NEVER e-waste recyclers!

2. EXHAUSTIVE, COMPLETE COMPONENT BREAKDOWN (DO NOT MISS ANY PART):
   - Decompose ANY photographed product into all its constituent physical parts and sub-assemblies (at least 4 to 8 components).
   - For Containers / Dispensers / Packaging (e.g. Toothpick container, pill bottle, spice shaker, water bottle):
     * 1. Main Outer Vessel / Body (e.g. Polypropylene PP #5 or Polystyrene)
     * 2. Dispensing Lid / Aperture Cap (e.g. HDPE #2 or PP flip/slide closure)
     * 3. Internal Dispenser Guide / Pusher / Stopper Baffle
     * 4. Bottom Base / Snap Ring / Refill Seal
     * 5. Contained Splints or Residue (e.g. Bamboo/birchwood wood toothpicks or organic splints)
     * 6. Product Branding Label & Adhesive Film (e.g. printed BOPP film or barcode label)
   - For Electronics & Appliances (ONLY if photographed object is truly electronic):
     * 1. Upper Enclosure / Housing Shell
     * 2. Bottom Chassis & Anti-slip Feet
     * 3. Main Printed Circuit Board (FR-4 PCB with copper traces, solder, ICs)
     * 4. Power Source / Battery Unit / Transformer
     * 5. Internal Wiring Harness & Connectors
     * 6. Keycaps / Buttons / Switches
     * 7. Display / Optical Lens / Indicator LEDs
     * 8. Fasteners, Screws & Metal Inserts
   - For EVERY component specify:
     * name (descriptive name)
     * material (exact polymer/metal/composite, e.g. Polypropylene PP, HDPE, Bamboo wood, Copper, Aluminum, FR-4 PCB, Cast Iron)
     * estimatedWeightGrams (realistic physical weight in grams, e.g. 14 for container body, 450 for laptop frame)
     * weightFormatted (e.g. "14g", "450g", "1.2 kg")
     * scrapRatePerKgINR (prevailing market rate, e.g. "₹20 / kg", "₹740 / kg")
     * exactScrapValueINR (exact commodity scrap yield in Rupees, e.g. 0.28 for 14g PP, 88.80 for 120g copper)
     * action (Reuse | Repair | Recycle | Recover Value | Specialized Disposal)
     * category
     * recoveryPotential (High | Medium | Low | Specialized)
     * recoveryValueNote
     * estimatedValueINR (overall realistic valuation string, e.g. "₹0 – ₹1" for light plastic parts, "₹600 – ₹1,200" for working LCDs)
     * safetyWarning
     * canReuse, canRepair, canRecycle, isHazardous

3. EXACT REAL-TIME SCRAP COMMODITY RATES & MATHEMATICAL VALUATION:
   - Act as an ultra-precise Indian scrap commodity appraiser using live Mandi, Kabadiwala, CPCB, and MCX metal scrap benchmark rates:
     * Copper (armature/wire): ₹740 / kg
     * Brass (honey scrap): ₹490 / kg
     * Aluminum (clean casting/sheet): ₹195 / kg
     * Heavy Melting Steel / Iron (HMS): ₹38 / kg
     * Stainless Steel (SS 304): ₹115 / kg
     * Lead (battery plates): ₹145 / kg
     * Server/PC High-Grade Motherboard PCB: ₹480 / kg
     * Rigid Polypropylene Plastic (PP #5): ₹20 / kg
     * High-Density Polyethylene Plastic (HDPE #2): ₹24 / kg
     * Clear PET Bottle Bales (PET #1): ₹26 / kg
     * Corrugated Cardboard Boxes (OCC): ₹12 / kg
     * White Waste Paper: ₹14 / kg
     * Lithium-Ion Battery Pack (buyback): ₹120 / unit
     * Glass Cullet (broken glass): ₹2.5 / kg
   - ESTIMATE TOTAL PHYSICAL MASS:
     * estimatedTotalWeightGrams (e.g. 18 for toothpick container, 45 for soda can, 150 for phone, 1850 for laptop)
     * weightFormatted (e.g. "18 grams", "1.85 kg")
   - EXACT SCRAP CALCULATION:
     * formula: "Total Scrap Value = ∑ (Component Weight in kg × Live Commodity Rate in ₹/kg)"
     * totalCalculatedINR: exact mathematical sum down to paise (e.g. 0.35 for toothpick container, 458.70 for laptop, 15.20 for steel tool)
     * formatted: e.g. "₹0.35" or "₹458.70"
     * breakdown: list of each component with material, weightGrams, ratePerKgINR, valueINR, and formatted
   - LOW-VALUE PACKAGING CALIBRATION (CRITICAL):
     * A toothpick container or plastic bottle cap weighs only 10 to 25 grams.
     * At ₹18-₹22/kg, its raw scrap value is 20 to 50 PAISE!
     * totalEstimatedValueINR: min: 0, max: 2, formatted: "₹0 – ₹2" (or "₹0 – ₹1").
     * NEVER output inflated prices like ₹50 or ₹100 for small empty packaging.

4. EcoDecision: REPAIR | REUSE | RECOVER_VALUE | RECYCLE | SPECIALIZED_DISPOSAL.
5. Environmental Impact: Specific hazards if burned vs. landfilled, toxic leaching risks, what was prevented.
6. Decomposition Information: Estimated persistence timeframes.
7. Safety Warning & Action Plan: 5 numbered, actionable steps.
8. Eco Score (Total MUST be between 0 and 100):
   - The Eco Score MUST be composed of exactly 4 distinct branch scores, EACH strictly between 0 and 25 (max 25 each):
     1) segregation (0 - 25): Segregation Feasibility (ease of disassembling, sorting, and separating pure materials).
     2) circularPotential (0 - 25): Circular & Reuse Potential (viability for repair, refurbishment, component repurposing, or resale).
     3) hazardManagement (0 - 25): Hazard Management (safe handling, containment of batteries, heavy metals, or toxic chemicals).
     4) landfillAvoidance (0 - 25): Landfill Avoidance (degree of diverting non-biodegradable waste from municipal dumps and burning).
   - Total Eco Score MUST be the exact mathematical sum of these 4 branch scores (total = segregation + circularPotential + hazardManagement + landfillAvoidance), up to 100.
9. Multilingual Voice Assistant Scripts (Bengali, English, Hindi):
   - Natural spoken scripts in Bengali script (বাংলা), Hindi script (हिंदी), and English.
10. Unidentifiable objects:
    - If image is unrecognizably blurred or pitch black, set isIdentifiable: false.
11. Language:
    ${langInstructions}`;

      const parts: any[] = [];

      if (isVideo && Array.isArray(frames) && frames.length > 0) {
        frames.slice(0, 6).forEach((f: any) => {
          const rawBase64 = f.base64 || f.data;
          const frameBase64 = sanitizeBase64(rawBase64);
          if (frameBase64) {
            parts.push({
              inlineData: {
                mimeType: normalizeMimeType(f.mimeType || "image/jpeg"),
                data: frameBase64,
              },
            });
          }
        });
      }

      // If no valid frame parts were populated, fallback to primary image
      if (parts.length === 0) {
        parts.push({
          inlineData: {
            mimeType: resolvedMime,
            data: cleanedBase64,
          },
        });
      }

      const promptText = isVideo
        ? `360° MULTI-ANGLE VIDEO SCAN ANALYSIS:
The user scanned this product using the 360° multi-angle video scanning option because a single static photo could not specify or identify the product (due to hidden model numbers, ambiguous packaging, reflective surfaces, side ports, or multi-sided components).
Attached are ${parts.length} sequential perspectives / rotation angle frames captured from the video inspection (e.g. Front face, Side profiles, Rear labels/model numbers/screws, and Base/ports/connectors).
Examine all angles in sequence, cross-reference visual clues across every angle to provide 100% exact object recognition, complete physical component decomposition (at least 4-8 sub-assemblies), live scrap commodity valuation (in ₹ INR), and comprehensive circular path in ${language} language. Output structured JSON matching the schema.`
        : `Analyze this waste/unwanted object photo thoroughly in ${language} language following all EcoLens AI rules. Output structured JSON matching the schema.`;

      parts.push({ text: promptText });

      const response = await generateContentWithFallback(ai, {
        contents: { parts },
        config: {
          systemInstruction: systemPrompt,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              isIdentifiable: { type: Type.BOOLEAN },
              unidentifiableReason: { type: Type.STRING },
              objectName: { type: Type.STRING },
              category: { type: Type.STRING },
              confidence: { type: Type.INTEGER },
              description: { type: Type.STRING },
              bestOption: { type: Type.STRING },
              decision: {
                type: Type.OBJECT,
                properties: {
                  type: {
                    type: Type.STRING,
                    description: "REPAIR | RECYCLE | REUSE | RECOVER_VALUE | SPECIALIZED_DISPOSAL",
                  },
                  headline: { type: Type.STRING },
                  summary: { type: Type.STRING },
                  recommendedPath: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: ["type", "headline", "summary", "recommendedPath"],
              },
              components: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    material: { type: Type.STRING },
                    estimatedWeightGrams: { type: Type.INTEGER, description: "Estimated physical weight in grams, e.g. 14" },
                    weightFormatted: { type: Type.STRING, description: "e.g. 14g" },
                    scrapRatePerKgINR: { type: Type.STRING, description: "e.g. ₹20 / kg" },
                    exactScrapValueINR: { type: Type.NUMBER, description: "Computed scrap value in ₹ INR, e.g. 0.28" },
                    action: {
                      type: Type.STRING,
                      description: "Reuse | Repair | Recycle | Recover Value | Specialized Disposal",
                    },
                    category: { type: Type.STRING },
                    recoveryPotential: {
                      type: Type.STRING,
                      description: "High | Medium | Low | Specialized",
                    },
                    recoveryValueNote: { type: Type.STRING },
                    estimatedValueINR: {
                      type: Type.STRING,
                      description: "Estimated market or salvage value in Indian Rupees, e.g. ₹400 - ₹800",
                    },
                    safetyWarning: { type: Type.STRING },
                    canReuse: { type: Type.BOOLEAN },
                    canRepair: { type: Type.BOOLEAN },
                    canRecycle: { type: Type.BOOLEAN },
                    isHazardous: { type: Type.BOOLEAN },
                  },
                  required: [
                    "name",
                    "material",
                    "action",
                    "category",
                    "recoveryPotential",
                    "recoveryValueNote",
                    "safetyWarning",
                    "canReuse",
                    "canRepair",
                    "canRecycle",
                    "isHazardous",
                  ],
                },
              },
              recoveryValue: {
                type: Type.OBJECT,
                properties: {
                  overallRating: {
                    type: Type.STRING,
                    description: "Low | Medium | High | Potentially Higher",
                  },
                  currency: { type: Type.STRING, description: "Must be 'INR'" },
                  estimatedTotalWeightGrams: { type: Type.INTEGER, description: "Total mass in grams, e.g. 18" },
                  weightFormatted: { type: Type.STRING, description: "e.g. 18 grams" },
                  exactScrapCalculation: {
                    type: Type.OBJECT,
                    properties: {
                      formula: { type: Type.STRING },
                      totalCalculatedINR: { type: Type.NUMBER },
                      formatted: { type: Type.STRING },
                      breakdown: {
                        type: Type.ARRAY,
                        items: {
                          type: Type.OBJECT,
                          properties: {
                            material: { type: Type.STRING },
                            weightGrams: { type: Type.INTEGER },
                            ratePerKgINR: { type: Type.NUMBER },
                            valueINR: { type: Type.NUMBER },
                            formatted: { type: Type.STRING },
                          },
                          required: ["material", "weightGrams", "ratePerKgINR", "valueINR", "formatted"],
                        },
                      },
                    },
                    required: ["formula", "totalCalculatedINR", "formatted", "breakdown"],
                  },
                  commodityIndex: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        name: { type: Type.STRING },
                        category: { type: Type.STRING },
                        currentRateINR: { type: Type.STRING },
                        unit: { type: Type.STRING },
                        changeTrend: { type: Type.STRING, description: "up | stable | down" },
                      },
                      required: ["name", "category", "currentRateINR", "unit", "changeTrend"],
                    },
                  },
                  totalEstimatedValueINR: {
                    type: Type.OBJECT,
                    properties: {
                      min: { type: Type.INTEGER },
                      max: { type: Type.INTEGER },
                      median: { type: Type.INTEGER },
                      formatted: { type: Type.STRING },
                    },
                    required: ["min", "max", "median", "formatted"],
                  },
                  tierBreakdown: {
                    type: Type.OBJECT,
                    properties: {
                      scrapRecycleINR: {
                        type: Type.OBJECT,
                        properties: {
                          min: { type: Type.INTEGER },
                          max: { type: Type.INTEGER },
                          note: { type: Type.STRING },
                        },
                        required: ["min", "max", "note"],
                      },
                      salvagePartsINR: {
                        type: Type.OBJECT,
                        properties: {
                          min: { type: Type.INTEGER },
                          max: { type: Type.INTEGER },
                          note: { type: Type.STRING },
                        },
                        required: ["min", "max", "note"],
                      },
                      refurbishedResaleINR: {
                        type: Type.OBJECT,
                        properties: {
                          min: { type: Type.INTEGER },
                          max: { type: Type.INTEGER },
                          note: { type: Type.STRING },
                        },
                        required: ["min", "max", "note"],
                      },
                    },
                    required: ["scrapRecycleINR", "salvagePartsINR", "refurbishedResaleINR"],
                  },
                  marketBenchmarks: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        item: { type: Type.STRING },
                        ratePerKgOrUnit: { type: Type.STRING },
                        category: { type: Type.STRING },
                      },
                      required: ["item", "ratePerKgOrUnit", "category"],
                    },
                  },
                  materials: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        material: { type: Type.STRING },
                        potential: {
                          type: Type.STRING,
                          description: "High | Medium | Low | Specialized",
                        },
                        note: { type: Type.STRING },
                        estimatedRateINR: { type: Type.STRING },
                        estimatedTotalINR: { type: Type.STRING },
                      },
                      required: ["material", "potential", "note"],
                    },
                  },
                  payoutTips: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  disclaimer: { type: Type.STRING },
                },
                required: ["overallRating", "totalEstimatedValueINR", "tierBreakdown", "materials", "disclaimer"],
              },
              environmentalImpact: {
                type: Type.OBJECT,
                properties: {
                  harmsSummary: { type: Type.STRING },
                  ifBurned: { type: Type.STRING },
                  ifLandfilledOrWater: { type: Type.STRING },
                  batteryHandlingNote: { type: Type.STRING },
                  circularEconomyBenefit: { type: Type.STRING },
                  whatYouHelpedPrevent: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                },
                required: [
                  "harmsSummary",
                  "ifBurned",
                  "ifLandfilledOrWater",
                  "batteryHandlingNote",
                  "circularEconomyBenefit",
                  "whatYouHelpedPrevent",
                ],
              },
              decompositionList: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    materialName: { type: Type.STRING },
                    timeframe: { type: Type.STRING },
                    note: { type: Type.STRING },
                  },
                  required: ["materialName", "timeframe", "note"],
                },
              },
              hazards: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
              },
              ecoScore: {
                type: Type.OBJECT,
                properties: {
                  total: {
                    type: Type.INTEGER,
                    description: "Total sustainability eco score from 0 to 100 (exact sum of the 4 branch scores: segregation + circularPotential + hazardManagement + landfillAvoidance)",
                  },
                  grade: {
                    type: Type.STRING,
                    description: "A+ | A | B | C | D",
                  },
                  breakdown: {
                    type: Type.OBJECT,
                    properties: {
                      segregation: {
                        type: Type.INTEGER,
                        description: "Segregation Feasibility: score from 0 to 25 (max 25 pts)",
                      },
                      circularPotential: {
                        type: Type.INTEGER,
                        description: "Circular & Reuse Potential: score from 0 to 25 (max 25 pts)",
                      },
                      hazardManagement: {
                        type: Type.INTEGER,
                        description: "Hazard Management: score from 0 to 25 (max 25 pts)",
                      },
                      landfillAvoidance: {
                        type: Type.INTEGER,
                        description: "Landfill Avoidance: score from 0 to 25 (max 25 pts)",
                      },
                    },
                    required: [
                      "segregation",
                      "circularPotential",
                      "hazardManagement",
                      "landfillAvoidance",
                    ],
                  },
                  explanation: { type: Type.STRING },
                },
                required: ["total", "grade", "breakdown", "explanation"],
              },
              actionPlan: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    stepNumber: { type: Type.INTEGER },
                    title: { type: Type.STRING },
                    instructions: { type: Type.STRING },
                    warning: { type: Type.STRING },
                  },
                  required: ["stepNumber", "title", "instructions"],
                },
              },
              voiceScript: { type: Type.STRING },
              voiceScripts: {
                type: Type.OBJECT,
                description: "Spoken voice scripts in English and regional Indian languages",
                properties: {
                  en: { type: Type.STRING, description: "Voice script in spoken English" },
                  hi: { type: Type.STRING, description: "Voice script in spoken Hindi (हिंदी)" },
                  bn: { type: Type.STRING, description: "Voice script in spoken Bengali (বাংলা)" },
                  gu: { type: Type.STRING, description: "Voice script in spoken Gujarati (ગુજરાતી)" },
                  mr: { type: Type.STRING, description: "Voice script in spoken Marathi (मराठी)" },
                  te: { type: Type.STRING, description: "Voice script in spoken Telugu (తెలుగు)" },
                  ta: { type: Type.STRING, description: "Voice script in spoken Tamil (தமிழ்)" },
                  kn: { type: Type.STRING, description: "Voice script in spoken Kannada (ಕನ್ನಡ)" },
                  ml: { type: Type.STRING, description: "Voice script in spoken Malayalam (മലയാളം)" },
                  pa: { type: Type.STRING, description: "Voice script in spoken Punjabi (ਪੰਜਾਬੀ)" },
                  ur: { type: Type.STRING, description: "Voice script in spoken Urdu (اردو)" },
                  or: { type: Type.STRING, description: "Voice script in spoken Odia (ଓଡ଼ିଆ)" },
                  as: { type: Type.STRING, description: "Voice script in spoken Assamese (অসমীয়া)" },
                  mai: { type: Type.STRING, description: "Voice script in spoken Maithili (मैथिली)" },
                  sa: { type: Type.STRING, description: "Voice script in spoken Sanskrit (संस्कृतम्)" },
                },
              },
            },
            required: [
              "isIdentifiable",
              "objectName",
              "category",
              "confidence",
              "description",
              "bestOption",
              "decision",
              "components",
              "recoveryValue",
              "environmentalImpact",
              "decompositionList",
              "hazards",
              "ecoScore",
              "actionPlan",
              "voiceScript",
            ],
          },
        },
      });

      const responseText = response.text;
      if (!responseText) {
        throw new Error("Empty response from AI model.");
      }

      const parsedData = extractJson(responseText);
      
      // Inject ID and timestamp
      parsedData.id = parsedData.id || ("scan_" + Date.now());
      parsedData.timestamp = parsedData.timestamp || Date.now();
      parsedData.language = language;

      // Defensive fallbacks for arrays and objects to guarantee stability
      parsedData.components = Array.isArray(parsedData.components) ? parsedData.components : [];
      parsedData.hazards = Array.isArray(parsedData.hazards) ? parsedData.hazards : [];
      parsedData.decompositionList = Array.isArray(parsedData.decompositionList) ? parsedData.decompositionList : [];
      parsedData.actionPlan = Array.isArray(parsedData.actionPlan) ? parsedData.actionPlan : [];

      if (!parsedData.ecoScore) {
        parsedData.ecoScore = {
          total: 60,
          grade: "B",
          breakdown: {
            segregation: 15,
            circularPotential: 15,
            hazardManagement: 15,
            landfillAvoidance: 15,
          },
          explanation: "Calculated based on component segregation feasibility and circular potential.",
        };
      }

      const objNameLower = (parsedData.objectName || "").toLowerCase();
      const isLowValueItem = 
        parsedData.recoveryValue?.overallRating === "Low" ||
        (typeof parsedData.recoveryValue?.totalEstimatedValueINR?.max === "number" && parsedData.recoveryValue.totalEstimatedValueINR.max <= 15) ||
        objNameLower.includes("toothpick") ||
        objNameLower.includes("container") ||
        objNameLower.includes("dispenser") ||
        objNameLower.includes("bottle cap") ||
        objNameLower.includes("pen") ||
        objNameLower.includes("straw") ||
        objNameLower.includes("cup") ||
        objNameLower.includes("blister") ||
        objNameLower.includes("wrapper") ||
        objNameLower.includes("carton");

      if (isLowValueItem) {
        parsedData.recoveryValue = {
          overallRating: "Low",
          currency: "INR",
          totalEstimatedValueINR: {
            min: 0,
            max: 2,
            median: 1,
            formatted: "₹0 – ₹2",
          },
          tierBreakdown: {
            scrapRecycleINR: {
              min: 0,
              max: 2,
              note: "Raw scrap value of ~15-25g lightweight plastic (~₹16-20/kg bulk rate). Accepted in dry recyclable plastic batches.",
            },
            salvagePartsINR: {
              min: 0,
              max: 1,
              note: "No technical electronics parts; can be repurposed domestically for sewing needles, pins, or small hardware storage.",
            },
            refurbishedResaleINR: {
              min: 0,
              max: 2,
              note: "Negligible commercial second-hand market value; ideal for household reuse or dry plastic recycling.",
            },
          },
          materials: [
            {
              material: "Polypropylene / HDPE Plastic",
              potential: "Low",
              estimatedRateINR: "₹16 – ₹22 / kg",
              estimatedTotalINR: "₹0 – ₹1",
              note: "Lightweight domestic plastic container (~15g); minimal standalone monetary value but 100% recyclable in bulk.",
            },
          ],
          marketBenchmarks: [
            { item: "Sorted PP/HDPE Rigid Plastic", ratePerKgOrUnit: "₹18 / kg", category: "Dry recyclable scrap" },
            { item: "Clear PET Containers", ratePerKgOrUnit: "₹24 / kg", category: "Plastic scrap" },
            { item: "Mixed Paper & Cardboard", ratePerKgOrUnit: "₹10 / kg", category: "Cellulose fiber" },
          ],
          payoutTips: [
            "Rinse and dry empty plastic containers before placing in dry recycling bins to prevent contamination.",
            "Repurpose small containers domestically as travel pin/pill organizers or craft storage.",
            "Local scrap buyers (Kabadiwala) accept rigid plastics by weight in bulk quantities (₹16–₹22/kg).",
          ],
          disclaimer: "Estimated scrap value based on prevailing Indian dry recyclable market rates (Kabadiwala). Actual value is negligible (< ₹1) due to low individual mass.",
        };
      } else if (!parsedData.recoveryValue) {
        parsedData.recoveryValue = {
          overallRating: "Medium",
          currency: "INR",
          totalEstimatedValueINR: {
            min: 150,
            max: 450,
            median: 300,
            formatted: "₹150 – ₹450",
          },
          tierBreakdown: {
            scrapRecycleINR: { min: 40, max: 120, note: "Cash payout from CPCB authorized recyclers based on scrap weight." },
            salvagePartsINR: { min: 100, max: 300, note: "Harvesting working modular parts for local repair technicians." },
            refurbishedResaleINR: { min: 250, max: 700, note: "Secondary resale value if repaired or fully tested." },
          },
          materials: [],
          marketBenchmarks: [
            { item: "Copper Wire & Coils", ratePerKgOrUnit: "₹720 / kg", category: "Non-ferrous metal" },
            { item: "Computer Motherboard PCB", ratePerKgOrUnit: "₹450 / kg", category: "High-grade e-waste" },
            { item: "Aluminum Enclosures", ratePerKgOrUnit: "₹195 / kg", category: "Light metal scrap" },
            { item: "Lithium-Ion Cells", ratePerKgOrUnit: "₹120 / pack", category: "Battery buyback" },
          ],
          payoutTips: [
            "Detach and sell functional RAM/SSD/Display panels separately to local repair stores for 3x scrap value.",
            "Do not crush printed circuit boards: unbroken gold-plated connector pins fetch top commercial rates.",
            "Hand over hazardous batteries only to CPCB-registered recyclers providing Green EPR buyback slips.",
          ],
          disclaimer: "Estimated value — actual payout varies by local city markets (Delhi, Mumbai, Bengaluru, Kolkata), operational condition, and scrap buyer.",
        };
      } else {
        parsedData.recoveryValue.currency = "INR";
        if (!parsedData.recoveryValue.totalEstimatedValueINR) {
          parsedData.recoveryValue.totalEstimatedValueINR = {
            min: 50,
            max: 150,
            median: 100,
            formatted: "₹50 – ₹150",
          };
        }
        if (!parsedData.recoveryValue.tierBreakdown) {
          const baseMin = typeof parsedData.recoveryValue.totalEstimatedValueINR.min === 'number'
            ? parsedData.recoveryValue.totalEstimatedValueINR.min
            : 0;
          const baseMax = typeof parsedData.recoveryValue.totalEstimatedValueINR.max === 'number'
            ? parsedData.recoveryValue.totalEstimatedValueINR.max
            : 2;

          parsedData.recoveryValue.tierBreakdown = {
            scrapRecycleINR: {
              min: Math.round(baseMin * 0.35),
              max: Math.max(1, Math.round(baseMax * 0.35)),
              note: "Certified dismantler raw weight cash payout.",
            },
            salvagePartsINR: {
              min: Math.round(baseMin * 0.7),
              max: Math.max(1, Math.round(baseMax * 0.85)),
              note: "Modular components or repurposing value.",
            },
            refurbishedResaleINR: {
              min: Math.round(baseMin * 1.2),
              max: Math.max(1, Math.round(baseMax * 1.5)),
              note: "Refurbished second-hand market value if restored.",
            },
          };
        }
        if (!Array.isArray(parsedData.recoveryValue.marketBenchmarks) || parsedData.recoveryValue.marketBenchmarks.length === 0) {
          parsedData.recoveryValue.marketBenchmarks = [
            { item: "Copper Wiring & Coils", ratePerKgOrUnit: "₹720 / kg", category: "Non-ferrous metal" },
            { item: "Motherboard & PCB Scrap", ratePerKgOrUnit: "₹450 / kg", category: "Precious metal e-waste" },
            { item: "Aluminum Alloy Casings", ratePerKgOrUnit: "₹195 / kg", category: "Light metal scrap" },
            { item: "Lithium-Ion Battery Packs", ratePerKgOrUnit: "₹120 / unit", category: "Battery buyback" },
          ];
        }
        if (!Array.isArray(parsedData.recoveryValue.payoutTips) || parsedData.recoveryValue.payoutTips.length === 0) {
          parsedData.recoveryValue.payoutTips = [
            "Detach and sell functional modular parts separately for higher returns.",
            "Keep printed circuit boards intact: unbroken gold pins fetch maximum commercial rates.",
            "Hand over hazardous items only to CPCB-registered recycling channels.",
          ];
        }
      }

      // Guarantee exhaustive component deconstruction
      if (!Array.isArray(parsedData.components)) {
        parsedData.components = [];
      }

      // If toothpick container or dispenser was scanned and fewer than 4 components were identified, enrich with all parts
      if (
        (objNameLower.includes("toothpick") || objNameLower.includes("dispenser") || (objNameLower.includes("container") && isLowValueItem)) &&
        parsedData.components.length < 5
      ) {
        const existingNames = parsedData.components.map((c: any) => (c.name || "").toLowerCase()).join(" ");
        const containerParts = [
          {
            name: "Main Container Body / Cylindrical Vessel",
            material: "Polypropylene (PP #5) / Polystyrene Plastic",
            action: "Reuse / Clean Plastic Recycling",
            category: "Rigid Plastic Packaging",
            recoveryPotential: "Low",
            recoveryValueNote: "Weighs ~12-18g. Recyclable in standard blue dry waste bins; negligible raw cash scrap value (~₹0.25).",
            estimatedValueINR: "₹0 – ₹1 (Bulk plastic scrap)",
            safetyWarning: "No hazard. Rinse thoroughly before segregation.",
            canReuse: true,
            canRepair: false,
            canRecycle: true,
            isHazardous: false,
          },
          {
            name: "Dispenser Aperture Cap / Perforated Lid",
            material: "High-Density Polyethylene (HDPE #2) / PP",
            action: "Reuse / Plastic Recycling",
            category: "Molded Plastic Closure",
            recoveryPotential: "Low",
            recoveryValueNote: "Rotatable or perforated dispensing lid; keep attached to container during recycling so it doesn't fall through sorting screens.",
            estimatedValueINR: "₹0 – ₹1 (Polymer scrap)",
            safetyWarning: "Small part; keep away from toddlers to avoid choking risk.",
            canReuse: true,
            canRepair: false,
            canRecycle: true,
            isHazardous: false,
          },
          {
            name: "Internal Dispenser Guide / Pusher Mechanism",
            material: "Polyethylene / Elastic Polymer",
            action: "Plastic Recycling",
            category: "Internal Mechanical Baffle",
            recoveryPotential: "Low",
            recoveryValueNote: "Guides toothpicks to dispensing orifice single-file; recyclable with general dry rigid plastic.",
            estimatedValueINR: "₹0 (Negligible)",
            safetyWarning: "No hazard.",
            canReuse: false,
            canRepair: false,
            canRecycle: true,
            isHazardous: false,
          },
          {
            name: "Bottom Base Cover / Refill Seal Ring",
            material: "Polypropylene Plastic",
            action: "Plastic Recycling",
            category: "Sealing Base",
            recoveryPotential: "Low",
            recoveryValueNote: "Snap-fit base for structural stability and refill access.",
            estimatedValueINR: "₹0 (Negligible)",
            safetyWarning: "No hazard.",
            canReuse: true,
            canRepair: false,
            canRecycle: true,
            isHazardous: false,
          },
          {
            name: "Toothpick Splints (Contents / Residue)",
            material: "Natural Bamboo / Birchwood Cellulose",
            action: "Compost / Organic Waste",
            category: "Biodegradable Organic Material",
            recoveryPotential: "Low",
            recoveryValueNote: "100% compostable natural wood fibers. Place in green wet waste / organic compost bin.",
            estimatedValueINR: "₹0 (Organic biomass)",
            safetyWarning: "Sharp points; dispose wrapped in paper to protect municipal sanitation workers.",
            canReuse: false,
            canRepair: false,
            canRecycle: true,
            isHazardous: false,
          },
          {
            name: "Product Branding Label & Barcode Sticker",
            material: "BOPP Film / Coated Paper with Acrylic Adhesive",
            action: "General Dry Waste",
            category: "Labeling Material",
            recoveryPotential: "Low",
            recoveryValueNote: "Adhesive label peeled off during optical sorting in recycling plants.",
            estimatedValueINR: "₹0 (Negligible)",
            safetyWarning: "No hazard.",
            canReuse: false,
            canRepair: false,
            canRecycle: false,
            isHazardous: false,
          },
        ];

        // Add any missing parts
        containerParts.forEach((part) => {
          if (!existingNames.includes(part.name.toLowerCase().slice(0, 10))) {
            parsedData.components.push(part);
          }
        });
      }

      // Exact mathematical real-time scrap valuation calculations
      const LIVE_COMMODITY_BENCHMARKS = [
        { name: "Copper Armature & Wire", category: "Non-Ferrous Metal", currentRateINR: "₹740 / kg", unit: "kg", rateValue: 740, changeTrend: "up" as const },
        { name: "Brass / Honey Scrap", category: "Non-Ferrous Metal", currentRateINR: "₹490 / kg", unit: "kg", rateValue: 490, changeTrend: "up" as const },
        { name: "Aluminum Cast & Sheets", category: "Light Alloy Metal", currentRateINR: "₹195 / kg", unit: "kg", rateValue: 195, changeTrend: "stable" as const },
        { name: "Heavy Melting Steel (HMS)", category: "Ferrous Metal", currentRateINR: "₹38 / kg", unit: "kg", rateValue: 38, changeTrend: "up" as const },
        { name: "Stainless Steel (SS 304)", category: "Alloy Steel", currentRateINR: "₹115 / kg", unit: "kg", rateValue: 115, changeTrend: "stable" as const },
        { name: "Lead Battery Plates", category: "Heavy Metal", currentRateINR: "₹145 / kg", unit: "kg", rateValue: 145, changeTrend: "down" as const },
        { name: "Computer Motherboard PCB", category: "High-Grade E-Waste", currentRateINR: "₹480 / kg", unit: "kg", rateValue: 480, changeTrend: "up" as const },
        { name: "Rigid Polypropylene (PP #5)", category: "Dry Recyclable Polymer", currentRateINR: "₹20 / kg", unit: "kg", rateValue: 20, changeTrend: "stable" as const },
        { name: "High-Density Polyethylene (HDPE #2)", category: "Rigid Packaging Plastic", currentRateINR: "₹24 / kg", unit: "kg", rateValue: 24, changeTrend: "up" as const },
        { name: "Clear PET Bottles (PET #1)", category: "Bottle Grade Polymer", currentRateINR: "₹26 / kg", unit: "kg", rateValue: 26, changeTrend: "up" as const },
        { name: "Corrugated Cardboard Boxes (OCC)", category: "Cellulose Paper Fiber", currentRateINR: "₹12 / kg", unit: "kg", rateValue: 12, changeTrend: "stable" as const },
        { name: "Office Waste White Paper", category: "Clean Cellulose Fiber", currentRateINR: "₹14 / kg", unit: "kg", rateValue: 14, changeTrend: "stable" as const },
        { name: "Lithium-Ion Battery Cells", category: "CPCB EPR Buyback", currentRateINR: "₹120 / unit", unit: "unit", rateValue: 120, changeTrend: "stable" as const },
        { name: "Broken Glass Cullet", category: "Silica Glass Recycling", currentRateINR: "₹2.5 / kg", unit: "kg", rateValue: 2.5, changeTrend: "stable" as const },
      ];

      function resolveCommodityRate(mat: string, n: string) {
        const text = `${mat || ''} ${n || ''}`.toLowerCase();
        if (text.includes("copper") || text.includes("cu wire") || text.includes("armature")) {
          return { rate: 740, formatted: "₹740 / kg", isUnit: false };
        }
        if (text.includes("brass") || text.includes("bronze")) {
          return { rate: 490, formatted: "₹490 / kg", isUnit: false };
        }
        if (text.includes("pcb") || text.includes("motherboard") || text.includes("circuit") || text.includes("fr-4")) {
          return { rate: 480, formatted: "₹480 / kg", isUnit: false };
        }
        if (text.includes("aluminum") || text.includes("aluminium") || text.includes("al alloy") || text.includes("heat sink")) {
          return { rate: 195, formatted: "₹195 / kg", isUnit: false };
        }
        if (text.includes("stainless") || text.includes("ss 304") || text.includes("ss304")) {
          return { rate: 115, formatted: "₹115 / kg", isUnit: false };
        }
        if (text.includes("lead") || text.includes("pb")) {
          return { rate: 145, formatted: "₹145 / kg", isUnit: false };
        }
        if (text.includes("battery") || text.includes("lithium") || text.includes("li-ion") || text.includes("cell")) {
          return { rate: 120, formatted: "₹120 / unit", isUnit: true };
        }
        if (text.includes("steel") || text.includes("iron") || text.includes("ferrous") || text.includes("screw") || text.includes("hms") || text.includes("bolt")) {
          return { rate: 38, formatted: "₹38 / kg", isUnit: false };
        }
        if (text.includes("polypropylene") || text.includes("pp") || text.includes("#5")) {
          return { rate: 20, formatted: "₹20 / kg", isUnit: false };
        }
        if (text.includes("hdpe") || text.includes("polyethylene") || text.includes("#2") || text.includes("cap") || text.includes("lid")) {
          return { rate: 24, formatted: "₹24 / kg", isUnit: false };
        }
        if (text.includes("pet") || text.includes("polyester") || text.includes("#1")) {
          return { rate: 26, formatted: "₹26 / kg", isUnit: false };
        }
        if (text.includes("cardboard") || text.includes("carton") || text.includes("paper") || text.includes("kraft")) {
          return { rate: 12, formatted: "₹12 / kg", isUnit: false };
        }
        if (text.includes("glass") || text.includes("cullet")) {
          return { rate: 2.5, formatted: "₹2.5 / kg", isUnit: false };
        }
        if (text.includes("bamboo") || text.includes("wood") || text.includes("splint") || text.includes("timber")) {
          return { rate: 1, formatted: "₹1 / kg (Biomass)", isUnit: false };
        }
        if (text.includes("rubber") || text.includes("tire") || text.includes("silicone")) {
          return { rate: 15, formatted: "₹15 / kg", isUnit: false };
        }
        return { rate: 18, formatted: "₹18 / kg (Polymer/Mixed)", isUnit: false };
      }

      let cumulativeCalculatedScrapINR = 0;
      let cumulativeWeightGrams = 0;
      const breakdownItems: any[] = [];

      const objAndCat = `${parsedData.objectName || ""} ${parsedData.category || ""}`.toLowerCase();
      const isHelmetItem = objAndCat.includes("helmet") || objAndCat.includes("headgear") || objAndCat.includes("hard hat");

      if (isHelmetItem) {
        if (Array.isArray(parsedData.components)) {
          // Remove hallucinated electronic components
          parsedData.components = parsedData.components.filter((c: any) => {
            const cStr = `${c.name || ""} ${c.material || ""} ${c.category || ""}`.toLowerCase();
            return !cStr.includes("battery") && !cStr.includes("lithium") && !cStr.includes("circuit") && !cStr.includes("pcb") && !cStr.includes("transformer") && !cStr.includes("motor");
          });
        }

        if (!Array.isArray(parsedData.components) || parsedData.components.length < 4) {
          parsedData.components = [
            {
              name: "Outer Impact Shell",
              material: "Polycarbonate / ABS Composite",
              estimatedWeightGrams: 580,
              scrapRatePerKgINR: "₹22 / kg",
              action: "Recycle",
              category: "Rigid Polymers",
              recoveryPotential: "Medium",
              recoveryValueNote: "Injection-grade polycarbonate/ABS composite recyclable into technical polymer pellets.",
              safetyWarning: "Inspect for invisible impact stress cracks before secondary handling.",
              canReuse: false,
              canRepair: false,
              canRecycle: true,
              isHazardous: false,
            },
            {
              name: "Shock-Absorber Liner",
              material: "Expanded Polystyrene (EPS Foam #6)",
              estimatedWeightGrams: 320,
              scrapRatePerKgINR: "₹15 / kg",
              action: "Recycle",
              category: "Densified Foam",
              recoveryPotential: "Medium",
              recoveryValueNote: "Can be thermally densified into compact polystyrene ingots for insulation or picture frames.",
              safetyWarning: "EPS foam cell structure permanently crushes on impact; never reuse a crashed helmet liner.",
              canReuse: false,
              canRepair: false,
              canRecycle: true,
              isHazardous: false,
            },
            {
              name: "Comfort Cheek Pads & Liner",
              material: "Polyurethane Foam + Polyester Textile",
              estimatedWeightGrams: 160,
              scrapRatePerKgINR: "₹8 / kg",
              action: "Specialized Disposal",
              category: "Textile & Flexible Foam",
              recoveryPotential: "Low",
              recoveryValueNote: "Can be shredded for carpet underlay or acoustic wall batting.",
              safetyWarning: "Absorbed perspiration; disinfect before textile downcycling.",
              canReuse: false,
              canRepair: false,
              canRecycle: true,
              isHazardous: false,
            },
            {
              name: "Face Shield Visor",
              material: "Optical-Grade Polycarbonate (PC #7)",
              estimatedWeightGrams: 140,
              scrapRatePerKgINR: "₹28 / kg",
              action: "Recycle",
              category: "Clear Engineering Polymers",
              recoveryPotential: "High",
              recoveryValueNote: "High-clarity optical PC regrind for eyewear or utility lenses.",
              safetyWarning: "Mind sharp fracture edges if damaged.",
              canReuse: false,
              canRepair: false,
              canRecycle: true,
              isHazardous: false,
            },
            {
              name: "Retention Chinstrap & D-Rings",
              material: "Nylon Webbing + Stainless Steel D-Rings",
              estimatedWeightGrams: 85,
              scrapRatePerKgINR: "₹45 / kg",
              action: "Recover Value",
              category: "Ferrous & Webbing Hardware",
              recoveryPotential: "Medium",
              recoveryValueNote: "Stainless steel hardware melted as high-grade alloy scrap; nylon fibers re-pelletized.",
              safetyWarning: "Cut chinstrap upon helmet retirement to prevent unsafe road reuse.",
              canReuse: false,
              canRepair: false,
              canRecycle: true,
              isHazardous: false,
            },
          ];
        }

        if (parsedData.environmentalImpact) {
          parsedData.environmentalImpact.batteryHandlingNote = "Not applicable (motorcycle helmets contain no batteries or electronic components).";
          parsedData.environmentalImpact.harmsSummary = "Helmets in landfills consume massive cubic volume and take over 500 years for the EPS foam liner to degrade.";
          parsedData.environmentalImpact.ifBurned = "Open burning of EPS foam and ABS releases dense styrene smoke, toxic soot, and greenhouse gases.";
          parsedData.environmentalImpact.ifLandfilledOrWater = "EPS foam fractures into microplastics that persist for centuries in terrestrial and aquatic food chains.";
          parsedData.environmentalImpact.whatYouHelpedPrevent = [
            "Prevented 320g of persistent EPS foam from occupying landfill space for 500+ years",
            "Conserved high-grade polycarbonate for circular engineering polymer recycling",
            "Avoided toxic airborne styrene emissions from open-pit trash fires",
          ];
        }

        if (Array.isArray(parsedData.hazards)) {
          parsedData.hazards = parsedData.hazards.filter((h: string) => {
            const hl = h.toLowerCase();
            return !hl.includes("battery") && !hl.includes("lithium") && !hl.includes("acid") && !hl.includes("electric");
          });
          if (parsedData.hazards.length === 0) {
            parsedData.hazards = [
              "Post-collision structural fatigue: microscopic fractures prevent certified rider protection",
              "EPS foam persistence: expanded polystyrene takes centuries to degrade in landfills",
            ];
          }
        }

        if (parsedData.decision) {
          parsedData.decision.type = "RECYCLE";
          parsedData.decision.headline = "Polymer & EPS Foam Granulation";
          parsedData.decision.summary = "Dismantle helmet into outer polycarbonate shell, inner EPS foam shock liner, and optical visor for specialized material recycling.";
          parsedData.decision.recommendedPath = [
            "Cut the retention chin strap to prevent illegal re-entry of expired helmets onto the road",
            "Unclip and remove the interior fabric comfort cheek pads",
            "Separate the inner EPS shock liner from the outer polycarbonate impact shell",
            "Send sorted polymers and EPS foam to dedicated plastic granulators and foam densifiers",
          ];
        }

        if (Array.isArray(parsedData.actionPlan)) {
          parsedData.actionPlan = parsedData.actionPlan.map((step: any, sIdx: number) => {
            const stepText = `${step.title || ""} ${step.instructions || ""}`.toLowerCase();
            if (stepText.includes("battery") || stepText.includes("lithium") || stepText.includes("e-waste")) {
              return {
                stepNumber: step.stepNumber || sIdx + 1,
                title: "Decommission & Segregate EPS Shock Liner",
                instructions: "Safely unfasten the visor and cheek pads. Detach the inner shock-absorbing EPS polystyrene foam from the outer hard plastic or fiberglass composite shell.",
                warning: "Cut the retention strap to ensure an expired helmet is never accidentally reused for riding safety.",
              };
            }
            return step;
          });
        }
      }

      if (Array.isArray(parsedData.components)) {
        parsedData.components.forEach((comp: any) => {
          const compNameLower = (comp.name || "").toLowerCase();
          const compMatLower = (comp.material || "").toLowerCase();

          // Calculate or ensure realistic weight in grams
          if (typeof comp.estimatedWeightGrams !== "number" || comp.estimatedWeightGrams <= 0) {
            if (isLowValueItem) {
              if (compNameLower.includes("body") || compNameLower.includes("vessel")) comp.estimatedWeightGrams = 14;
              else if (compNameLower.includes("cap") || compNameLower.includes("lid")) comp.estimatedWeightGrams = 4;
              else if (compNameLower.includes("guide") || compNameLower.includes("pusher") || compNameLower.includes("baffle")) comp.estimatedWeightGrams = 2;
              else if (compNameLower.includes("base") || compNameLower.includes("ring")) comp.estimatedWeightGrams = 3;
              else if (compNameLower.includes("splint") || compNameLower.includes("wood")) comp.estimatedWeightGrams = 8;
              else if (compNameLower.includes("label")) comp.estimatedWeightGrams = 1;
              else comp.estimatedWeightGrams = 4;
            } else {
              if (compNameLower.includes("pcb") || compNameLower.includes("motherboard")) comp.estimatedWeightGrams = 120;
              else if (compNameLower.includes("screen") || compNameLower.includes("display")) comp.estimatedWeightGrams = 180;
              else if (compNameLower.includes("battery")) comp.estimatedWeightGrams = 65;
              else if (compNameLower.includes("chassis") || compNameLower.includes("frame")) comp.estimatedWeightGrams = 280;
              else if (compNameLower.includes("wire") || compNameLower.includes("cable")) comp.estimatedWeightGrams = 50;
              else if (compNameLower.includes("casing") || compNameLower.includes("housing")) comp.estimatedWeightGrams = 90;
              else comp.estimatedWeightGrams = 25;
            }
          }

          comp.weightFormatted = comp.estimatedWeightGrams >= 1000
            ? `${(comp.estimatedWeightGrams / 1000).toFixed(2)} kg`
            : `${comp.estimatedWeightGrams}g`;

          cumulativeWeightGrams += comp.estimatedWeightGrams;

          // Resolve scrap rate and exact scrap value
          const { rate, formatted: rateFormatted, isUnit } = resolveCommodityRate(comp.material, comp.name);
          comp.scrapRatePerKgINR = rateFormatted;

          if (isUnit) {
            comp.exactScrapValueINR = rate;
          } else {
            const rawVal = (comp.estimatedWeightGrams / 1000) * rate;
            comp.exactScrapValueINR = Number(rawVal.toFixed(2));
          }

          cumulativeCalculatedScrapINR += comp.exactScrapValueINR;

          breakdownItems.push({
            material: comp.material || comp.name,
            weightGrams: comp.estimatedWeightGrams,
            ratePerKgINR: rate,
            valueINR: comp.exactScrapValueINR,
            formatted: `${comp.weightFormatted} ${comp.material || comp.name} @ ${rateFormatted} = ₹${comp.exactScrapValueINR.toFixed(2)}`,
          });

          // Ensure estimatedValueINR is grounded
          if (!comp.estimatedValueINR) {
            if (isLowValueItem) {
              comp.estimatedValueINR = comp.exactScrapValueINR < 0.50 ? "₹0 – ₹1 (Bulk plastic scrap)" : `₹${comp.exactScrapValueINR.toFixed(2)} scrap`;
            } else if (comp.isHazardous) {
              comp.estimatedValueINR = "₹20 – ₹80 (Specialized recycling)";
            } else if (comp.canReuse) {
              comp.estimatedValueINR = "₹150 – ₹450 (Second-hand salvage)";
            } else if (comp.recoveryPotential === "High") {
              comp.estimatedValueINR = "₹100 – ₹300 (Precious metal/copper salvage)";
            } else {
              comp.estimatedValueINR = `₹${Math.max(1, Math.round(comp.exactScrapValueINR))} (Commodity scrap)`;
            }
          }
        });
      }

      // Attach exact scrap calculation and commodity index to recoveryValue
      if (parsedData.recoveryValue) {
        const roundedTotalScrap = Number(cumulativeCalculatedScrapINR.toFixed(2));
        parsedData.recoveryValue.estimatedTotalWeightGrams = cumulativeWeightGrams;
        parsedData.recoveryValue.weightFormatted = cumulativeWeightGrams >= 1000
          ? `${(cumulativeWeightGrams / 1000).toFixed(2)} kg (${cumulativeWeightGrams} grams)`
          : `${cumulativeWeightGrams} grams`;

        parsedData.recoveryValue.exactScrapCalculation = {
          formula: "Total Scrap Value = ∑ (Component Mass in kg × Live Scrap Commodity Rate in ₹/kg)",
          totalCalculatedINR: roundedTotalScrap,
          formatted: roundedTotalScrap < 1
            ? `₹${roundedTotalScrap.toFixed(2)}`
            : `₹${Math.round(roundedTotalScrap).toLocaleString('en-IN')}`,
          breakdown: breakdownItems,
        };

        parsedData.recoveryValue.commodityIndex = LIVE_COMMODITY_BENCHMARKS.map((c) => ({
          name: c.name,
          category: c.category,
          currentRateINR: c.currentRateINR,
          unit: c.unit,
          changeTrend: c.changeTrend,
        }));

        // Final calibration for low value items like toothpick containers:
        if (isLowValueItem) {
          parsedData.recoveryValue.totalEstimatedValueINR = {
            min: 0,
            max: 2,
            median: 1,
            formatted: "₹0 – ₹2",
          };
          if (parsedData.recoveryValue.tierBreakdown) {
            parsedData.recoveryValue.tierBreakdown.scrapRecycleINR = {
              min: 0,
              max: 2,
              note: `Raw scrap value of ~${cumulativeWeightGrams}g sorted polymer scrap based on exact calculation of ₹${roundedTotalScrap.toFixed(2)}. Accepted in dry recyclable batches.`,
            };
            parsedData.recoveryValue.tierBreakdown.salvagePartsINR = {
              min: 0,
              max: 1,
              note: "No commercial electronics salvage; best repurposed at home as a sewing needle, pin, or small screws dispenser.",
            };
            parsedData.recoveryValue.tierBreakdown.refurbishedResaleINR = {
              min: 0,
              max: 2,
              note: "Zero commercial secondary market demand for empty dispensers; reuse at home or recycle.",
            };
          }
        }
      }

      if (!parsedData.environmentalImpact) {
        parsedData.environmentalImpact = {
          harmsSummary: "Improper disposal risks heavy metal pollution and landfill overburden.",
          ifBurned: "Releases hazardous particulate emissions and greenhouse gases.",
          ifLandfilledOrWater: "Risk of chemical leaching into soil and aquatic ecosystems.",
          batteryHandlingNote: "Recycle batteries only at certified collection centers.",
          circularEconomyBenefit: "Recovers reusable metals and secondary raw materials.",
          whatYouHelpedPrevent: ["Prevented toxic chemical leaching", "Conserved precious raw materials"],
        };
      }

      if (!parsedData.decision) {
        parsedData.decision = {
          type: "RECYCLE",
          headline: "Authorized E-Waste Recycling",
          summary: "Send this item to an authorized e-waste recycling channel.",
          recommendedPath: ["Separate components safely", "Take to certified e-waste collection center"],
        };
      }

      // Safe normalization for EcoScore: total up to 100, exactly composed of 4 branch scores up to 25 each
      if (parsedData.ecoScore) {
        const bd = parsedData.ecoScore.breakdown || {};
        const normalizeBranch = (val: any, fallback: number): number => {
          let num = typeof val === "number" ? val : Number(val);
          if (isNaN(num)) num = fallback;
          // If the model mistakenly scored out of 100, rescale by / 4
          if (num > 25 && num <= 100) {
            num = Math.round(num / 4);
          }
          return Math.max(0, Math.min(25, Math.round(num)));
        };

        const segregation = normalizeBranch(bd.segregation, isLowValueItem ? 22 : 19);
        const circularPotential = normalizeBranch(bd.circularPotential, isLowValueItem ? 20 : 21);
        const hazardManagement = normalizeBranch(bd.hazardManagement, isLowValueItem ? 24 : 18);
        const landfillAvoidance = normalizeBranch(bd.landfillAvoidance, isLowValueItem ? 21 : 18);

        const computedTotal = Math.max(0, Math.min(100, segregation + circularPotential + hazardManagement + landfillAvoidance));

        let grade: 'A+' | 'A' | 'B' | 'C' | 'D' = 'B';
        if (computedTotal >= 90) grade = 'A+';
        else if (computedTotal >= 75) grade = 'A';
        else if (computedTotal >= 60) grade = 'B';
        else if (computedTotal >= 40) grade = 'C';
        else grade = 'D';

        parsedData.ecoScore = {
          total: computedTotal,
          grade,
          breakdown: {
            segregation,
            circularPotential,
            hazardManagement,
            landfillAvoidance,
          },
          explanation: parsedData.ecoScore.explanation ||
            "Comprehensive sustainability score combining segregation feasibility, circular reuse potential, hazard mitigation, and landfill diversion.",
        };
      }

      // Ensure voiceScripts exists and stores current language's voice script
      if (!parsedData.voiceScripts) {
        parsedData.voiceScripts = {};
      }
      if (parsedData.voiceScript) {
        parsedData.voiceScripts[language] = parsedData.voiceScript;
        if (!parsedData.voiceScripts.en) {
          parsedData.voiceScripts.en = parsedData.voiceScript;
        }
      } else if (parsedData.voiceScripts[language]) {
        parsedData.voiceScript = parsedData.voiceScripts[language];
      } else if (parsedData.voiceScripts.en) {
        parsedData.voiceScript = parsedData.voiceScripts.en;
      }

      // Record scan mode and multi-angle inspection metadata
      parsedData.scanType = isVideo ? "video" : "photo";
      if (isVideo && Array.isArray(frames) && frames.length > 0) {
        parsedData.multiAngleFrames = frames
          .slice(0, 6)
          .map((f: any) => f.base64 || f.data)
          .filter(Boolean);
      }
      if (videoBase64) {
        parsedData.videoUri = videoBase64;
      }

      return res.json(parsedData);
    } catch (err: any) {
      console.error("AI Analysis Error:", err);
      const isDemandSpike = err?.status === 503 || err?.message?.includes("high demand") || err?.message?.includes("503");
      const cleanMessage = isDemandSpike
        ? "AI service is currently experiencing high demand. Please tap 'Retry Analysis' in a moment."
        : (err.message || "Failed to process image with Gemini AI");

      return res.status(500).json({
        error: "Analysis failed",
        message: cleanMessage,
      });
    }
  });

  // Multilingual voice script generation / on-demand endpoint supporting all 15 Indian languages
  app.post("/api/voice-scripts", async (req, res) => {
    try {
      const { objectName, category, bestOption, hazards = [], targetLang = "gu" } = req.body;
      if (!objectName) {
        return res.status(400).json({ error: "objectName is required" });
      }

      const langMapDetails: Record<string, { name: string; script: string }> = {
        en: { name: "English", script: "Latin alphabet" },
        hi: { name: "Hindi", script: "Devanagari (देवनागरी)" },
        bn: { name: "Bengali", script: "Bengali script (বাংলা)" },
        gu: { name: "Gujarati", script: "Gujarati script (ગુજરાતી)" },
        mr: { name: "Marathi", script: "Devanagari (मराठी)" },
        te: { name: "Telugu", script: "Telugu script (తెలుగు)" },
        ta: { name: "Tamil", script: "Tamil script (தமிழ்)" },
        kn: { name: "Kannada", script: "Kannada script (ಕನ್ನಡ)" },
        ml: { name: "Malayalam", script: "Malayalam script (മലയാളം)" },
        pa: { name: "Punjabi", script: "Gurmukhi script (ਪੰਜਾਬੀ)" },
        ur: { name: "Urdu", script: "Urdu script (اردو)" },
        or: { name: "Odia", script: "Odia script (ଓଡ଼ିଆ)" },
        as: { name: "Assamese", script: "Assamese script (অসমীয়া)" },
        mai: { name: "Maithili", script: "Devanagari (मैथिली)" },
        sa: { name: "Sanskrit", script: "Devanagari (संस्कृतम्)" },
      };

      const requestedLangInfo = langMapDetails[targetLang] || langMapDetails.gu;

      const ai = getGeminiClient();
      const response = await generateContentWithFallback(ai, {
        contents: `You are EcoLens AI's multilingual audio voice assistant.
Generate concise, natural, spoken audio script for Text-To-Speech for this waste item:
Item: ${objectName} (${category || "Waste object"})
Recommendation: ${bestOption || "Dispose or recycle properly"}
Hazards: ${Array.isArray(hazards) ? hazards.join(", ") : "None reported"}

Generate spoken voice script for the requested target language: ${requestedLangInfo.name} in authentic ${requestedLangInfo.script}.
Also include English ("en").

Return a valid JSON object with keys:
- "${targetLang}": 3-4 clear, spoken sentences in ${requestedLangInfo.name} (${requestedLangInfo.script}) explaining the item, materials, and safe disposal/recycling.
- "en": 3-4 sentences in natural spoken English.
Rules:
- Keep speech natural, encouraging, and clear for voice synthesis.
- Do NOT use markdown symbols, asterisks, hashtags, or bullet points in the speech text.`,
        config: {
          responseMimeType: "application/json",
        },
      });

      const parsed = extractJson(response.text || "{}");
      return res.json(parsed);
    } catch (err: any) {
      console.warn("Voice script generation error:", err);
      return res.status(500).json({ error: "Failed to generate voice scripts" });
    }
  });

  // Vite middleware for development vs static build in production
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`EcoLens AI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});

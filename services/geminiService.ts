import { GoogleGenAI, Type } from "@google/genai";
import type { LiveServerMessage } from "@google/genai";

// Initialize Gemini Client
// NOTE: process.env.API_KEY is injected by the environment.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- 1. Thinking Mode (Complex Business Advice) ---
export const getBusinessAdvice = async (query: string): Promise<string> => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: query,
      config: {
        systemInstruction: "شما یک مشاور زیبایی هوشمند، بسیار با تجربه و محترم برای سالن زیبایی تخصصی زیباسافت هستید. به زبان فارسی بسیار روان، مودبانه و گرم صحبت می‌کنید. شما تخصص کاملی در زمینه مشاوره پوست و مو، استایل مو، میکاپ، کاشت ناخن، روتین‌های مراقبتی و ترکیب رنگ مو دارید. پاسخ‌های شما باید کاربردی، دقیق، علمی و الهام‌بخش باشد و لحنی صمیمانه اما حرفه‌ای داشته باشد. در پاسخ‌های خود از شکلک‌های مناسب زیبایی مانند ✨💅💇‍♀️🌸 نیز استفاده کنید.",
      },
    });
    return response.text || "متاسفانه پاسخی دریافت نشد.";
  } catch (error) {
    console.error("Thinking API Error:", error);
    return "خطایی در دریافت مشاوره هوشمند رخ داد. لطفا مجدد تلاش کنید.";
  }
};

// --- 2. Maps Grounding (Supplier/Location Search) ---
export interface MapResult {
  title: string;
  uri: string;
}

export const searchLocations = async (query: string, userLat?: number, userLng?: number): Promise<{ text: string, links: MapResult[] }> => {
  try {
    const config: any = {
      tools: [{ googleMaps: {} }],
    };

    // Add retrieval config if location is provided
    if (userLat && userLng) {
      config.toolConfig = {
        retrievalConfig: {
          latLng: {
            latitude: userLat,
            longitude: userLng
          }
        }
      };
    }

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: query,
      config: config
    });

    const text = response.text || "";
    
    // Extract map links from grounding chunks
    const links: MapResult[] = [];
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    
    if (chunks) {
      chunks.forEach((chunk: any) => {
        if (chunk.web?.uri && chunk.web?.title) {
             links.push({ title: chunk.web.title, uri: chunk.web.uri });
        }
        // Check specifically for maps chunks if the structure differs in future
        if (chunk.maps?.uri && chunk.maps?.title) {
            links.push({ title: chunk.maps.title, uri: chunk.maps.uri });
        }
      });
    }

    return { text, links };

  } catch (error) {
    console.error("Maps API Error:", error);
    return { text: "خطا در برقراری ارتباط با نقشه گوگل.", links: [] };
  }
};

// --- 3. Image Editing (Nano Banana) ---
export const editImageStyle = async (base64Image: string, mimeType: string, prompt: string): Promise<string | null> => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite-image",
      contents: {
        parts: [
          {
            inlineData: {
              data: base64Image,
              mimeType: mimeType
            }
          },
          {
            text: prompt
          }
        ]
      }
    });

    // Iterate to find image part
    if (response.candidates?.[0]?.content?.parts) {
      for (const part of response.candidates[0].content.parts) {
        if (part.inlineData && part.inlineData.data) {
          return `data:image/png;base64,${part.inlineData.data}`;
        }
      }
    }
    return null;
  } catch (error) {
    console.error("Image Edit Error:", error);
    throw error;
  }
};

// --- 4. Live API (Voice Agent) Utils ---

export const connectLiveSession = async (
  onOpen: () => void,
  onMessage: (message: LiveServerMessage) => void,
  onError: (e: ErrorEvent) => void,
  onClose: (e: CloseEvent) => void
) => {
  return ai.live.connect({
    model: 'gemini-3.1-flash-live-preview',
    callbacks: {
      onopen: onOpen,
      onmessage: onMessage,
      onerror: onError,
      onclose: onClose,
    },
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } }, // Friendly voice
      },
      systemInstruction: 'You are a helpful, polite, and professional AI receptionist for a high-end beauty salon in Iran named "ZibaSoft". You speak fluent Persian (Farsi). You help with scheduling, answering questions about services, and providing beauty tips. Be concise.',
    },
  });
};

// Helper to convert Float32 AudioBuffer to PCM Int16 Blob
export function createBlob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  
  // Custom encode function since Buffer/btoa might vary in environment
  let binary = '';
  const bytes = new Uint8Array(int16.buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  
  return {
    data: btoa(binary),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// Helper to decode Base64 to AudioBuffer
export async function decodeAudioData(
  base64Data: string,
  ctx: AudioContext,
  sampleRate: number = 24000,
  numChannels: number = 1
): Promise<AudioBuffer> {
  const binaryString = atob(base64Data);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const dataInt16 = new Int16Array(bytes.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}
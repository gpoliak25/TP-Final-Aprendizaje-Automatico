import { NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  const apiKey = process.env.ANTHROPIC_API_KEY
  if (!apiKey) {
    return NextResponse.json(
      { error: "ANTHROPIC_API_KEY no configurada en .env.local" },
      { status: 500 },
    )
  }
  const model = process.env.ANTHROPIC_MODEL ?? "claude-sonnet-4-6"

  const { imageBase64, mediaType, prob, predicted, actual } = await req.json()

  const contextLines = [
    `- Predicción del modelo: **${predicted}**`,
    `- Probabilidad de patología: **${Math.round(prob * 100)}%**`,
    actual ? `- Etiqueta real del dataset: **${actual}**` : null,
  ]
    .filter(Boolean)
    .join("\n")

  const prompt = `Sos un veterinario radiólogo especializado en diagnóstico por imágenes de pequeños animales.

Se te presenta una radiografía de tórax analizada por un modelo de deep learning (Transfer Learning MobileNetV2 entrenado en dataset veterinario):

${contextLines}

Analizá la imagen y respondé en español, en formato Markdown, usando EXACTAMENTE estos títulos de sección con "## " (en este orden):

## Conclusión diagnóstica
Resumen ejecutivo de 2 a 4 oraciones que sintetice el diagnóstico más probable y el nivel de confianza, integrando los hallazgos radiológicos con la predicción del modelo. Debe ser el PRIMER punto del informe.

## Hallazgos radiológicos
Describí las estructuras y hallazgos visibles (silueta cardíaca, pulmones, caja torácica, diafragma, etc.).

## Concordancia con el modelo
¿Los hallazgos radiológicos concuerdan con la predicción del modelo? Explicá el razonamiento.

## Diagnósticos diferenciales
Listá los posibles diagnósticos ordenados por probabilidad.

## Recomendaciones
Próximos pasos clínicos sugeridos.

No agregues un título general al principio: empezá directamente con "## Conclusión diagnóstica".

---
*Análisis demostrativo — Trabajo Final de Maestría en Ciencia de Datos, CAECE 2026.*`

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 3000,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType || "image/jpeg",
                data: imageBase64,
              },
            },
            { type: "text", text: prompt },
          ],
        },
      ],
    }),
  })

  if (!response.ok) {
    const err = await response.text()
    console.error("Claude API error:", err)
    return NextResponse.json(
      { error: `Claude API respondió ${response.status}` },
      { status: response.status },
    )
  }

  const data = await response.json()
  return NextResponse.json({ diagnosis: data.content[0].text })
}

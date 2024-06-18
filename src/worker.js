import {
  env,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  AutoProcessor,
  RawImage,
  CLIPVisionModelWithProjection,
} from "@xenova/transformers";
import imageEmbeddingJson from "./data/image-embedding-json.js";

const EMBED_DIM = 512;

// Skip local model check
env.allowLocalModels = false;
env.useBrowserCache = false;

class ApplicationSingleton {
  static model_id = "Xenova/clip-vit-base-patch16";

  static tokenizer = null;
  static text_model = null;
  static metadata = null;
  static embeddings = null;

  static async getInstance(progress_callback = null) {
    // Load tokenizer and text model
    if (this.tokenizer === null) {
      this.tokenizer = AutoTokenizer.from_pretrained(this.model_id, {
        progress_callback,
      });
    }
    if (this.text_model === null) {
      this.text_model = CLIPTextModelWithProjection.from_pretrained(
        this.model_id,
        { progress_callback }
      );
    }
    if (this.metadata === null) {
      this.metadata = imageEmbeddingJson;
    }
    if (this.embeddings === null) {
      let imageProcessor = AutoProcessor.from_pretrained(this.model_id);
      let visionModel = CLIPVisionModelWithProjection.from_pretrained(
        this.model_id,
        { quantized: true }
      );

      this.imageProcessor = imageProcessor;
      this.visionModel = visionModel;

      //   this.embeddings = result;
    }

    return Promise.all([
      this.tokenizer,
      this.text_model,
      this.metadata,
      this.embeddings,
      this.visionModel,
      this.imageProcessor,
    ]);
  }
}

function cosineSimilarity(query_embeds, database_embeds) {
  const numDB = database_embeds.length / EMBED_DIM;
  const similarityScores = new Array(numDB);

  for (let i = 0; i < numDB; ++i) {
    const startOffset = i * EMBED_DIM;
    const dbVector = database_embeds.slice(
      startOffset,
      startOffset + EMBED_DIM
    );

    let dotProduct = 0;
    let normEmbeds = 0;
    let normDB = 0;

    for (let j = 0; j < EMBED_DIM; ++j) {
      const embedValue = query_embeds[j];
      const dbValue = dbVector[j];

      dotProduct += embedValue * dbValue;
      normEmbeds += embedValue * embedValue;
      normDB += dbValue * dbValue;
    }

    similarityScores[i] =
      dotProduct / (Math.sqrt(normEmbeds) * Math.sqrt(normDB));
  }

  return similarityScores;
}

// Listen for messages from the main thread
self.addEventListener("message", async (event) => {
  // Get the tokenizer, model, metadata, and embeddings. When called for the first time,
  // this will load the files and cache them for future use.
  const [
    tokenizer,
    text_model,
    metadata,
    embeddings,
    imageProcessor,
    visionModel,
  ] = await ApplicationSingleton.getInstance(self.postMessage);

  // Send the output back to the main thread
  self.postMessage({ status: "ready" });

  // Run tokenization
  const text_inputs = tokenizer(event.data.text, {
    padding: true,
    truncation: true,
  });

  // Compute embeddings
  const { text_embeds } = await text_model(text_inputs);

  const result = imageEmbeddingJson.map(async (data) => {
    let image = await RawImage.read(data.url);
    let imageInputs = await imageProcessor(image);
    let { image_embeds } = await visionModel(imageInputs);
    return image_embeds;
  });

  console.log({
    metadata,
    result,
  });

  // Compute similarity scores
  const scores = cosineSimilarity(text_embeds.data, result);

  // Make a copy of the metadata
  let output = metadata.slice(0);

  // Add scores to output
  for (let i = 0; i < metadata.length; ++i) {
    output[i].score = scores[i];
  }

  // Sort by score
  output.sort((a, b) => b.score - a.score);

  // Get top 100 results
  output = output.slice(0, 100);

  // Send the output back to the main thread
  self.postMessage({
    status: "complete",
    output: output,
  });
});

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// Directory containing the images
const imagesDir = path.join(process.cwd(), 'src/images');

// Output JavaScript file
const outputFilePath = path.join(process.cwd(), 'src/data/image-embedding-json.js');

// Function to generate JSON containing image names
const generateImageList = async () => {
  try {
    const files = await fs.promises.readdir(imagesDir);

    // Filter out non-image files (optional)
    const imageFiles = files.filter((file) => {
      const ext = path.extname(file).toLowerCase();
      return ['.jpg', '.jpeg', '.png'].includes(ext);
    });

    // Create the JSON array
    const imageList = imageFiles.map((file) => ({
      id: uuidv4(),
      url: `/images/${file}`,
    }));

    // Create the content for the JavaScript file
    const jsContent = `const imageEmbeddingJson = ${JSON.stringify(imageList, null, 2)};\nexport default imageEmbeddingJson;`;

    // Write the JavaScript content to a file
    await fs.promises.writeFile(outputFilePath, jsContent);
    console.log('Image list generated successfully:', outputFilePath);
  } catch (err) {
    console.error('Error:', err);
  }
};

// Run the function
generateImageList();

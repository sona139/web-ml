import * as fs from 'fs';

// Function to read a binary file and process its contents
function readBinaryFile(filename) {
    fs.readFile(filename, (err, data) => {
        if (err) {
            console.error(`Error reading file ${filename}:`, err);
            return;
        }

        // Convert the entire data buffer to Float32Array
        const floatArray = new Float32Array(data.buffer);

        const jsContent = `const imageEmbeddingVector = ${JSON.stringify(floatArray, null, 2)};\nexport default imageEmbeddingVector;`;
        const outputFilePath = 'src/data/image-embedding-vectors.js'
        // Write the JavaScript content to a file
        fs.writeFileSync(outputFilePath, jsContent);

        // Now you can use floatArray for further processing
        console.log('Float32Array:', floatArray);
    });
}

// Helper function to convert bytes to hexadecimal string
function byteToHex(byte) {
    return ('0' + byte.toString(16)).slice(-2);
}

// Function to create a hexadecimal dump of the first `length` bytes of a Buffer
function getHexDump(buffer, length) {
    let hexDump = '';

    for (let i = 0; i < length; i++) {
        if (i % 16 === 0 && i !== 0) {
            hexDump += '\n';
        }
        const byte = buffer[i];
        hexDump += byteToHex(byte) + ' ';
    }

    return hexDump;
}

// Example usage: Replace 'filename.bin' with the path to your binary file
const filename = 'src/assets/stored_images.bin';
readBinaryFile(filename);

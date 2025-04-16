const imageUpload = document.getElementById('imageUpload');
const loadedMessage = document.getElementById('loadedMessage'); // Reference the loaded message

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(() => {
    // Show the "Assets Loaded" message once models are loaded
    loadedMessage.style.display = 'block';
    start(); // Start the face recognition process
});

async function start() {
    const resultsContainer = document.getElementById('results'); // Use the predefined image area container

    // Load Labeled Face Descriptors and log them
    const LabeledFaceDescriptors = await loadLabeledImages();
    console.log('Loaded Labeled Face Descriptors:', LabeledFaceDescriptors);

    // Use a lower threshold to allow more matches (0.4)
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors, 0.4);

    let image;
    let canvas;

    imageUpload.addEventListener('change', async () => {
        // Clear any existing image or canvas
        if (image) image.remove();
        if (canvas) canvas.remove();

        // Load and display the uploaded image
        image = await faceapi.bufferToImage(imageUpload.files[0]);
        resultsContainer.append(image);

        // Create and position the canvas correctly
        canvas = faceapi.createCanvasFromMedia(image);
        resultsContainer.append(canvas);

        // Ensure the canvas is on top of the image and aligned correctly
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.zIndex = '1';  // Ensure canvas is on top of the image

        const displaySize = { width: image.width, height: image.height };
        faceapi.matchDimensions(canvas, displaySize);

        // Detect faces and log the detections
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        console.log('Detected Faces:', detections);

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Log the descriptors being generated
        console.log('Generated Descriptors:', resizedDetections.map(d => d.descriptor));

        // Get the best matches for the detected faces
        const results = resizedDetections.map(d => {
            const bestMatch = faceMatcher.findBestMatch(d.descriptor);
            console.log(`Best match for face ${d.detection.box} :`, bestMatch); // Log the match result
            return bestMatch;
        });

        // Draw boxes and labels based on the best matches
        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
            drawBox.draw(canvas);
        });
    });
}

async function loadLabeledImages() {
    const labels = ['abdul', 'aryan', 'ashesh', 'deep', 'dharmik', 'harsh', 'kevin', 'kunj', 'om', 'parth', 'rugved', 'urvesh', 'vasu'];
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];

            for (let i = 1; i <= 2; i++) {
                try {
                    const img = await faceapi.fetchImage(`labeled_images/${label}/${i}.jpg`);
                    const detection = await faceapi
                        .detectSingleFace(img)
                        .withFaceLandmarks()
                        .withFaceDescriptor();

                    if (detection) {
                        descriptions.push(detection.descriptor);
                    } else {
                        console.warn(`No face detected in ${label}/${i}.jpg`);
                    }
                } catch (err) {
                    console.error(`Error loading image: ${label}/${i}.jpg`, err);
                }
            }

            // Log the descriptors for each label
            if (descriptions.length > 0) {
                console.log(`Descriptors for label "${label}":`, descriptions);
                return new faceapi.LabeledFaceDescriptors(label, descriptions);
            } else {
                console.warn(`Skipping label "${label}" - no valid face data`);
                return null;
            }
        })
    ).then(data => data.filter(x => x !== null)); // remove nulls
}

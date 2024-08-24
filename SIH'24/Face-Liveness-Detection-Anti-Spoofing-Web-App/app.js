const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();
}

function captureFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageData;
}

async function start() {
    await setupWebcam();  // Initialize webcam feed

    setInterval(() => {
        const frame = captureFrame();
        // Further processing can be done here
    }, 100); // Capture and process frames every 100ms
}

start()
function fetchNewsFromURL() {
    let url = document.getElementById("news-url").value;
    if (!url) {
        document.getElementById("error-message").innerText = "Please enter a URL.";
        return;
    }

    document.getElementById("error-message").innerText = "";
    document.getElementById("loading").style.display = "block";

    fetch("http://localhost:5000/fetch-news", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").style.display = "none";
        if (data.error) {
            document.getElementById("error-message").innerText = data.error;
        } else {
            document.getElementById("newsInput").value = data.content;
        }
    })
    .catch(error => {
        document.getElementById("error-message").innerText = "Failed to fetch news.";
        console.error("Error:", error);
    });
}

function checkNews() {
    let text = document.getElementById("newsInput").value.trim();

    if (!text) {
        document.getElementById("result").innerText = "Please enter some text!";
        return;
    }

    document.getElementById("result").innerText = "Checking...";

    fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Response from Server:", data);

        let prediction = data.prediction;
        let confidence = data.confidence || "N/A";

        // ✅ Redirect to result.html with data in URL parameters
        let encodedText = encodeURIComponent(text);
        let encodedPrediction = encodeURIComponent(prediction);
        let encodedConfidence = encodeURIComponent(confidence);

        window.location.href = `result.html?text=${encodedText}&prediction=${encodedPrediction}&confidence=${encodedConfidence}`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error checking news. Please try again.";
    });
}

function fetchNewsAndPredict() {
    fetchNewsFromURL();
    setTimeout(checkNews, 100); // Give some time for news content to load before prediction
}

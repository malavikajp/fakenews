function fetchNewsFromURL() {
    return new Promise((resolve, reject) => {
        let url = document.getElementById("news-url").value;
        if (!url) {
            document.getElementById("error-message").innerText = "Please enter a URL.";
            return reject("❌ No URL provided.");
        }

        document.getElementById("error-message").innerText = "";
        document.getElementById("loading").style.display = "block";

        if (!navigator.onLine) {
            document.getElementById("loading").style.display = "none";
            document.getElementById("error-message").innerText = "No internet connection.";
            return reject("❌ No internet connection.");
        }

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
                return reject(data.error);
            }

            if (data.language && data.language !== "en") {
                document.getElementById("error-message").innerText = "Error: Only English articles are supported.";
                return reject("❌ Non-English article.");
            }

            document.getElementById("newsInput").value = data.content;
            resolve();  // ✅ Resolve once news is fetched
        })
        .catch(error => {
            document.getElementById("loading").style.display = "none";
            document.getElementById("error-message").innerText = "Failed to fetch news.";
            console.error("Error:", error);
            reject(error);
        });
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

        console.log("Redirecting to result page...");
        window.location.href = `result.html?text=${encodedText}&prediction=${encodedPrediction}&confidence=${encodedConfidence}`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error checking news.";
    });
}

function fetchNewsAndPredict() {
    fetchNewsFromURL()
        .then(() => {
            console.log("News fetched, now checking...");
            checkNews();  // ✅ Ensure this runs after fetching
        })
        .catch(error => console.error("Fetching failed:", error));
}

// ✅ Hide elements and fetch news before prediction
function hide() {
    document.querySelectorAll(".hideclick").forEach(element => {
        element.style.display = "none";
    });
    fetchNewsAndPredict();
}

// ✅ Ensure event listener is properly attached
window.onload = function() {
    document.getElementById("b").addEventListener("click", hide);
};

//enter-btn Function
console.log("script.js loaded!");
document.getElementById("enter-btn").addEventListener("click", async () => {
    const enterBtn = document.getElementById("enter-btn"); 
    const outputSection = document.getElementById("article-output");

    // Retrieve selected quarter
    const quarter = document.querySelector('input[name="quarter"]:checked')?.id;
    const symbol = "PTT";

    // Retrieve input data
    const instruction = document.getElementById("instruction").value.trim();
    const data = document.getElementById("data").value.trim();
    const event = document.getElementById("event").value.trim();

    if (!quarter || !instruction || !data || !event) {
        alert("โปรดกรอกข้อมูลให้ครบทุกช่อง");
        return;
    }

    const BACKEND_URL = "/api/generate";
    const RESULT_URL = "/api/result";

    outputSection.innerHTML = '<div id="loader" class="loader"></div>';
    const loader = document.getElementById("loader");
    loader.style.opacity = "1";
    loader.style.visibility = "visible";

    enterBtn.disabled = true;
    enterBtn.style.opacity = "0.6";

    const articleOP = document.getElementById("article-output");
    articleOP.style.display = "flex";
    articleOP.style.justifyContent = "center";
    articleOP.style.alignItems = "center";
    articleOP.style.minHeight = "150px";
    articleOP.style.textAlign = "center";

    try {
        // Send request to start processing
        const input = { instruction, data, event };
        const params = new URLSearchParams({ symbol, quarter });
        const response = await fetch(`${BACKEND_URL}?${params}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(input)
        });

        if (!response.ok) throw new Error("เกิดข้อผิดพลาดในการโหลดข้อมูล");

        const { task_id } = await response.json();

        // Poll for result every 5 seconds
        async function pollResult() {
            const res = await fetch(`${RESULT_URL}?task_id=${task_id}`);
            const result = await res.json();

            if (result.status === "Processing") {
                setTimeout(pollResult, 5000);
            } else {
                loader.style.opacity = "0";
                loader.style.visibility = "hidden";
                loader.style.display = "none";

                articleOP.style.display = "";
                articleOP.style.justifyContent = "";
                articleOP.style.alignItems = "";
                articleOP.style.minHeight = "";
                articleOP.style.textAlign = "";
    
                enterBtn.disabled = false;
                enterBtn.style.opacity = "1.0";

                outputSection.innerHTML = `<p>${result.message}</p>`;
            }
        }

        pollResult();

    } catch (error) {
        console.error("Error:", error);
        alert("เกิดข้อผิดพลาด!");
        loader.style.opacity = "0";
        loader.style.visibility = "hidden";
        loader.style.display = "none";
    }
});

//copy-btn Function
document.getElementById("copy-btn").addEventListener("click", () => {
    const outputText = document.getElementById("article-output").textContent.trim();
    const copyBtn = document.getElementById("copy-btn");

    if (!outputText) {
        alert("ไม่มีเนื้อหาให้คัดลอก!");
        return;
    }

    navigator.clipboard.writeText(outputText)
        .then(() => {
            copyBtn.textContent = "Complete"; 
            setTimeout(() => {
                copyBtn.textContent = "Copy Article";
            }, 1000);
        })
        .catch(err => {
            console.error("Error copying text: ", err);
            alert("เกิดข้อผิดพลาดในการคัดลอก");
        });
});
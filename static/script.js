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

    const TIMEOUT = 100000;

    const BACKEND_URL = "/api/generate";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);

    if (!quarter || !instruction.trim() || !data.trim() || !event.trim()) {
        alert("โปรดกรอกข้อมูลให้ครบทุกช่อง");
        return;
    }
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
        const input = { instruction, data, event };
        const params = new URLSearchParams({ symbol, quarter });
        console.log("value to backend:", { input, params:params.toString() });

        const response = await fetch(`${BACKEND_URL}?${params}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify(input)
        });

        clearTimeout(timeoutId);

        if (!response.ok) throw new Error("เกิดข้อผิดพลาดในการโหลดข้อมูล");

        const result = await response.json();

        loader.style.opacity = "0";
        setTimeout(() => {
            loader.style.visibility = "hidden";
            loader.style.display = "none";
        }, 300);

        outputSection.innerHTML = `<p>${result.message}</p>`;

    } catch (error) {
        console.error("Error:", error);
        alert("⏳ Generate ใช้เวลาเกิน 100 วินาที หรือเกิดข้อผิดพลาด!");
        loader.style.opacity = "0";
        loader.style.visibility = "hidden";
        loader.style.display = "none";
    }
    articleOP.style.display = "";
    articleOP.style.justifyContent = "";
    articleOP.style.alignItems = "";
    articleOP.style.minHeight = "";
    articleOP.style.textAlign = "";
    
    enterBtn.disabled = false;
    enterBtn.style.opacity = "1.0";
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




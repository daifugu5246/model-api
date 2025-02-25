//enter-btn Function
document.getElementById("enter-btn").addEventListener("click", async () => {
    // Retrieve selected quarter
    const quarter = document.querySelector('input[name="quarter"]:checked')?.id;
    const symbol = "PTT";
    // Retrieve input data
    const instruction = document.getElementById("instruction").value.trim();
    const data = document.getElementById("data").value.trim();
    const event = document.getElementById("event").value.trim();

    const BACKEND_URL = "";

    if (!quarter || !instruction.trim() || !data.trim() || !event.trim()) {
        alert("โปรดกรอกข้อมูลให้ครบทุกช่อง");
        return;
    }
    const input = {
        instruction: instruction,
        data: data,
        event: event
    };

    // กำหนดค่าพารามิเตอร์
    const params = {
        symbol: symbol,
        quarter: quarter
    };
    
    console.log("value to backend:", { input, params });

    try {
        // ส่งข้อมูลไปยัง backend โดยใช้ Axios
        const response = await axios.post(BACKEND_URL, input, { params });

        // ตรวจสอบ response
        console.log("Response จาก Backend:", response.data);

        // แสดงผลลัพธ์บนเว็บ
        document.getElementById("article-output").textContent = JSON.stringify(response.data, null, 2);
    } catch (error) {
        console.error("Error sending data:", error);
        alert("เกิดข้อผิดพลาดในการส่งข้อมูล กรุณาลองใหม่อีกครั้ง");
    }

    // // Prepare payload for ChatGPT
    // const payload = {
    //     model: "gpt-4",
    //     messages: [
    //         { role: "system", content: "You are an AI that generates financial analysis articles." },
    //         { role: "user", content: `Quarter: ${quarter}\nInstruction: ${instruction}\nData: ${data}\nEvent: ${event}` }
    //     ]
    // };

    // // Send data to ChatGPT and fetch the generated analysis
    // try {
    //     const response = await fetch("https://api.openai.com/v1/chat/completions", {
    //         method: "POST",
    //         headers: {
    //             "Content-Type": "application/json",
    //             "Authorization": `Bearer YOUR_API_KEY_HERE`
    //         },
    //         body: JSON.stringify(payload)
    //     });

    //     if (!response.ok) {
    //         throw new Error(`HTTP error! status: ${response.status}`);
    //     }

    //     const result = await response.json();

    //     // Display the generated article
    //     document.getElementById("article-output").textContent = result.choices[0].message.content;
    // } catch (error) {
    //     console.error("Error generating article:", error);
    //     document.getElementById("article-output").textContent = "Error generating article. Please try again later.";
    // }
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




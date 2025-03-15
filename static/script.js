//enter-btn Function
document.getElementById("enter-btn").addEventListener("click", async () => {
    // Retrieve selected quarter
    const quarter = document.querySelector('input[name="quarter"]:checked')?.id;
    const symbol = "PTT";
    // Retrieve input data
    const instruction = document.getElementById("instruction").value.trim();
    const data = document.getElementById("data").value.trim();
    const event = document.getElementById("event").value.trim();

    const BACKEND_URL = "/api/generate";

    if (!quarter || !instruction.trim() || !data.trim() || !event.trim()) {
        alert("โปรดกรอกข้อมูลให้ครบทุกช่อง");
        return;
    }
    if (document.getElementById("article-output").textContent != '') {
        document.getElementById("article-output").textContent = '';
    }
    const input = {
        instruction: instruction,
        data: data,
        event: event
    };

    // กำหนดค่าพารามิเตอร์
    const params = new URLSearchParams ({
        symbol,
        quarter
    });
    
    console.log("value to backend:", { input, params:params.toString() });

    try {
        // ส่งข้อมูลไปยัง backend โดยใช้ Axios
        const response = await axios.post(`${BACKEND_URL}?${params.toString()}`, input, {headers: {"Content-Type" :"application/json"}, timeout : 600000});

        // ตรวจสอบ response
        console.log("Response จาก Backend:", response.data);

        // แสดงผลลัพธ์บนเว็บ
        document.getElementById("article-output").textContent = response.data.message;
    } catch (error) {
        console.error("Error sending data:", error);
        alert("เกิดข้อผิดพลาดในการส่งข้อมูล กรุณาลองใหม่อีกครั้ง");
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




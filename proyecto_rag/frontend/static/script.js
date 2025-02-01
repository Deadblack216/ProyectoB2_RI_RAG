document.getElementById("queryForm").addEventListener("submit", function (e) {
    e.preventDefault();
    const query = document.getElementById("queryInput").value;

    fetch("/query", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query }),
    })
    .then((response) => response.json())
    .then((data) => {
        document.getElementById("responseText").innerText = data.response;

        const documentList = document.getElementById("documentList");
        documentList.innerHTML = "";
        data.documents.forEach((doc) => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>Candidato:</strong> ${doc.Candidato}<br>
                            <strong>Temas tratados:</strong> ${doc["Temas tratados"]}<br>
                            <strong>Descripción:</strong> ${doc.Descripción}`;
            documentList.appendChild(li);
        });
    })
    .catch((error) => {
        console.error("Error:", error);
    });
});
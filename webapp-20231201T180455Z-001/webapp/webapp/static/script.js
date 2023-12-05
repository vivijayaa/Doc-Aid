
let symptoms = [];

document.getElementById('processButton').addEventListener('click', function() {
    const text = document.getElementById('inputText').value;
    processData(text);
});

document.getElementById('yesMoreSymptoms').addEventListener('click', function() {
    document.getElementById('symptomInputSection').style.display = 'block';
});

document.getElementById('addSymptom').addEventListener('click', function() {
    const symptom = document.getElementById('symptomInput').value;
    if (symptom) {
        symptoms.push(symptom);
        document.getElementById('symptomInput').value = ''; // Clear input field
        // updateSymptomsList(symptom); // Update the displayed list of symptoms
    }
});



document.getElementById('noMoreSymptoms').addEventListener('click', function() {
    document.getElementById('symptomInputSection').style.display = 'none';
    document.getElementById('symptomQuerySection').style.display = 'none';
    finalizeSymptoms(symptoms);
    displaySymptomsWithSeverity(symptoms)
});


function finalizeSymptoms(symptomsList) {


    fetch('/finalize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms: symptomsList })
    })
    .then(response => response.json())
    .then(symptoms => {
        displaySymptomsWithSeverity(symptoms);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

let uniqueSymptoms=[]
function displaySymptomsWithSeverity(symptoms) {
    let table = '<table><tr><th>Symptom</th><th>Severity</th></tr>';
    uniqueSymptoms = [...new Set(symptoms)];
    uniqueSymptoms.forEach((symptom, index) => {
        table += `<tr>
                    <td>${symptom}</td>
                    <td>
                        <select id="severity_${index}">
                            <option value="3">High</option>
                            <option value="2">Moderate</option>
                            <option value="1">Normal</option>
                        </select>
                    </td>
                  </tr>`;
    });
    table += '</table>';
    table += '<button onclick="submitSeverity()">Submit Severity</button>';
    document.getElementById('outputTable').innerHTML = table;
}

function submitSeverity() {
    const symptomSeverity = uniqueSymptoms.map((symptom, index) => ({
        symptom: symptom,
        severity: document.getElementById(`severity_${index}`).value
    }));

    sendSymptomSeverity(symptomSeverity);
}




function processData(text) {
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        displayTable(data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function displayTable(data) {
    // ... existing code to display table ...
    if (!Array.isArray(data) || data.length === 0) {
        document.getElementById('outputTable').innerHTML = 'No data to display';
        return;
    }

    let table = '<table>';
    table += '<tr>';
    for (let key in data[0]) {
        table += `<th>${key}</th>`;
    }
    table += '</tr>';

    data.forEach(row => {
        table += '<tr>';
        for (let key in row) {
            table += `<td>${row[key]}</td>`;
        }
        table += '</tr>';
    });
    table += '</table>';

    document.getElementById('outputTable').innerHTML = table;
    document.getElementById('symptomQuerySection').style.display = 'block';
}





function updateSymptomsList() {
    const symptomsElement = document.getElementById('symptomsList');
    let table = '<table><tr><th>Symptom</th><th>Severity</th></tr>';

    symptoms.forEach((symptom, index) => {
        table += `
            <tr>
                <td>${symptom}</td>
                <td>
                    <select id="severity_${index}">
                        <option value="high">High</option>
                        <option value="moderate">Moderate</option>
                        <option value="normal">Normal</option>
                    </select>
                </td>
            </tr>`;
    });

    table += '</table>';
    table += '<button id="submitSymptoms">Submit Symptoms</button>';

    symptomsElement.innerHTML = table;

    document.getElementById('submitSymptoms').addEventListener('click', submitSymptoms);
}
function submitSymptoms() {
    const symptomsSeverity = symptoms.map((symptom, index) => {
        return {
            symptom: symptom,
            severity: document.getElementById(`severity_${index}`).value
        };
    });

    sendSymptomsSeverityToBackend(symptomsSeverity);
}



function sendSymptomsSeverityToBackend(symptomsSeverity) {
    fetch('/process_symptoms_severity', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptomsSeverity })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Symptoms with severity processed:', data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}




// function sendSymptomSeverity(symptomSeverity) {
//     fetch('/process_symptoms_severity', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ symptomsSeverity: symptomSeverity })
//     })
//     .then(response => response.json())
//     .then(data => {
//         console.log('Symptoms with severity processed:', data);
//     })
//     .catch((error) => {
//         console.error('Error:', error);
//     });
// }
function sendSymptomSeverity(symptomSeverity) {
    fetch('/process_symptoms_severity', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptomsSeverity: symptomSeverity })
    })
    .then(response => response.json())
    .then(data => {
        displayDiseases(data.diseases);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function displayDiseases(diseases) {
    let html = '<p>These are your possible diseases:</p>';
    html += '<table>';
    diseases.forEach(disease => {
        html += `<tr><td>${disease}</td></tr>`;
    });
    html += '</table>';
    document.getElementById('outputTable').innerHTML = html;
}

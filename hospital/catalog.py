"""Catálogo sintético. `seed.py` persiste só a versão anonimizada."""
from __future__ import annotations

PROTOCOLS = [
    {
        "id": "PROT-STATIN-CABG",
        "title": "Preoperative statin before CABG",
        "pmid": "17625060",
        "keywords": "statin cabg atrial fibrillation preoperative",
        "body": (
            "Hospital protocol PROT-STATIN-CABG. Question: Do preoperative statins "
            "reduce atrial fibrillation after coronary artery bypass grafting? "
            "Evidence summary (PMID 17625060): perioperative statin use is associated "
            "with a lower rate of postoperative atrial fibrillation. Suggested conduct: "
            "review home statin, do not interrupt without cardiology review, flag if "
            "the patient is statin-naive before elective CABG. REQUIRES PHYSICIAN VALIDATION. "
            "Do not prescribe a dose in this assistant."
        ),
    },
    {
        "id": "PROT-PPI-UGIB",
        "title": "PPI before endoscopy for upper GI bleeding",
        "pmid": "18973770",
        "keywords": "ppi proton pump inhibitor endoscopy gastrointestinal bleeding",
        "body": (
            "Hospital protocol PROT-PPI-UGIB. High-dose proton pump inhibitor before "
            "endoscopy may reduce the need for endoscopic therapy in upper GI bleeding "
            "(PMID 18973770, evidence mixed — label maybe when trials conflict). "
            "Suggested conduct: ensure IV access, type and screen, urgent endoscopy pathway. "
            "REQUIRES PHYSICIAN VALIDATION. Do not start a drug from this assistant."
        ),
    },
    {
        "id": "PROT-STERID-CROUP",
        "title": "Glucocorticoids in croup",
        "pmid": "10591357",
        "keywords": "croup glucocorticoid steroid dexamethasone",
        "body": (
            "Hospital protocol PROT-STERID-CROUP. Glucocorticoids reduce symptoms and "
            "return visits in croup (PMID 10591357). Suggested conduct: assess airway, "
            "Westley score, oxygen saturation, escalate if stridor at rest. "
            "REQUIRES PHYSICIAN VALIDATION."
        ),
    },
    {
        "id": "PROT-ASPIRIN-PREECL",
        "title": "Low-dose aspirin for preeclampsia prevention",
        "pmid": "25098060",
        "keywords": "aspirin preeclampsia pregnancy prevention",
        "body": (
            "Hospital protocol PROT-ASPIRIN-PREECL. Low-dose aspirin started in the "
            "second trimester reduces preeclampsia in high-risk pregnancy (PMID 25098060). "
            "Suggested conduct: confirm risk factors in the chart, obstetric review. "
            "REQUIRES PHYSICIAN VALIDATION. Never prescribe from this assistant."
        ),
    },
    {
        "id": "PROT-ABX-DIVERT",
        "title": "Antibiotics in uncomplicated diverticulitis",
        "pmid": "22290281",
        "keywords": "antibiotics diverticulitis uncomplicated",
        "body": (
            "Hospital protocol PROT-ABX-DIVERT. Antibiotics may not improve recovery "
            "in CT-confirmed uncomplicated diverticulitis (PMID 22290281). Suggested "
            "conduct: confirm imaging, pain control, oral intake, return precautions. "
            "REQUIRES PHYSICIAN VALIDATION."
        ),
    },
    {
        "id": "PROT-TROPONIN-ACS",
        "title": "Serial troponin in suspected ACS",
        "pmid": "25173350",
        "keywords": "troponin acs chest pain serial biomarker",
        "body": (
            "Hospital protocol PROT-TROPONIN-ACS. A single negative troponin does not "
            "exclude ACS. Repeat high-sensitivity troponin per pathway (PMID 25173350). "
            "Suggested conduct: if chest pain and first troponin pending or negative, "
            "keep the patient monitored and emit an alert for the pending exam. "
            "REQUIRES PHYSICIAN VALIDATION."
        ),
    },
    {
        "id": "PROT-SEPSIS-LACTATE",
        "title": "Lactate and sepsis bundle",
        "pmid": "26903338",
        "keywords": "sepsis lactate bundle shock",
        "body": (
            "Hospital protocol PROT-SEPSIS-LACTATE. Elevated lactate with suspected "
            "infection triggers the sepsis bundle review (PMID 26903338). Suggested "
            "conduct: check pending cultures and lactate, alert the team, do not "
            "order antimicrobials from this assistant. REQUIRES PHYSICIAN VALIDATION."
        ),
    },
    {
        "id": "PROT-DVT-PROPH",
        "title": "VTE prophylaxis in medical inpatients",
        "pmid": "16549822",
        "keywords": "vte dvt prophylaxis heparin inpatient",
        "body": (
            "Hospital protocol PROT-DVT-PROPH. Pharmacologic VTE prophylaxis reduces "
            "events in at-risk medical inpatients (PMID 16549822) unless bleeding risk "
            "is high. Suggested conduct: review bleeding risk and mobility in the chart. "
            "REQUIRES PHYSICIAN VALIDATION. Do not prescribe heparin here."
        ),
    },
    {
        "id": "PROT-REPORT-TEMPLATE",
        "title": "Internal report and prescription template",
        "pmid": "17625060",
        "keywords": "report prescription procedure template validation",
        "body": (
            "Hospital template for reports, suggested prescriptions and procedures. "
            "Every generated document MUST end with REQUIRES PHYSICIAN VALIDATION. "
            "The assistant may draft a report citing PMID and protocol id. It must "
            "never issue a direct prescription, dose, route or administration order."
        ),
    },
]

RAW_PATIENTS = [
    {
        "name": "Ana Beatriz Lima",
        "cpf": "390.533.447-05",
        "chart_number": "PR-10021",
        "birth_date": "1968-04-12",
        "sex": "F",
        "phone": "(11) 98821-4401",
        "email": "ana.lima@email.test",
        "address": "Rua das Acacias 120, Sao Paulo",
        "allergies": "penicillin",
        "diagnosis": "Elective CABG planned. Statin-naive.",
        "notes": "Contact Ana Beatriz Lima, CPF 390.533.447-05, phone (11) 98821-4401, email ana.lima@email.test, Rua das Acacias 120.",
        "exams": [
            {"name": "ECG", "status": "completed", "result": "sinus rhythm", "ordered_at": "2026-03-01", "resulted_at": "2026-03-01"},
            {"name": "Lipid panel", "status": "pending", "result": "", "ordered_at": "2026-03-02", "resulted_at": ""},
        ],
    },
    {
        "name": "Carlos Eduardo Nunes",
        "cpf": "153.509.460-56",
        "chart_number": "PR-10022",
        "birth_date": "1955-11-03",
        "sex": "M",
        "phone": "(21) 99710-2208",
        "email": "carlos.nunes@email.test",
        "address": "Avenida Atlantica 900, Rio de Janeiro",
        "allergies": "none",
        "diagnosis": "Chest pain, suspected ACS.",
        "notes": "Carlos Eduardo Nunes CPF 153.509.460-56 lives at Avenida Atlantica 900. Phone (21) 99710-2208.",
        "exams": [
            {"name": "Troponin", "status": "pending", "result": "", "ordered_at": "2026-03-10", "resulted_at": ""},
            {"name": "ECG", "status": "completed", "result": "nonspecific ST changes", "ordered_at": "2026-03-10", "resulted_at": "2026-03-10"},
        ],
    },
    {
        "name": "Fernanda Souza",
        "cpf": "231.002.999-00",
        "chart_number": "PR-10023",
        "birth_date": "1992-07-19",
        "sex": "F",
        "phone": "(31) 98400-1190",
        "email": "fernanda.souza@email.test",
        "address": "Rua da Bahia 45, Belo Horizonte",
        "allergies": "none",
        "diagnosis": "High-risk pregnancy, preeclampsia screening.",
        "notes": "Fernanda Souza, email fernanda.souza@email.test, Rua da Bahia 45.",
        "exams": [
            {"name": "Blood pressure log", "status": "completed", "result": "148/92", "ordered_at": "2026-02-20", "resulted_at": "2026-02-20"},
            {"name": "Urine protein", "status": "pending", "result": "", "ordered_at": "2026-02-21", "resulted_at": ""},
        ],
    },
    {
        "name": "Joao Pedro Almeida",
        "cpf": "111.444.777-35",
        "chart_number": "PR-10024",
        "birth_date": "1979-01-30",
        "sex": "M",
        "phone": "(41) 99912-3344",
        "email": "joao.almeida@email.test",
        "address": "Rua XV de Novembro 10, Curitiba",
        "allergies": "sulfa",
        "diagnosis": "Uncomplicated diverticulitis on CT.",
        "notes": "Allergy to sulfa. Joao Pedro Almeida CPF 111.444.777-35.",
        "exams": [
            {"name": "CT abdomen", "status": "completed", "result": "uncomplicated diverticulitis", "ordered_at": "2026-01-15", "resulted_at": "2026-01-15"},
        ],
    },
    {
        "name": "Marina Costa",
        "cpf": "529.982.247-25",
        "chart_number": "PR-10025",
        "birth_date": "2019-09-08",
        "sex": "F",
        "phone": "(51) 98111-0099",
        "email": "marina.costa.guardian@email.test",
        "address": "Avenida Ipiranga 300, Porto Alegre",
        "allergies": "none",
        "diagnosis": "Croup, barking cough.",
        "notes": "Guardian phone (51) 98111-0099, Avenida Ipiranga 300.",
        "exams": [
            {"name": "SpO2", "status": "completed", "result": "96% room air", "ordered_at": "2026-04-02", "resulted_at": "2026-04-02"},
        ],
    },
    {
        "name": "Ricardo Mendes",
        "cpf": "862.883.667-57",
        "chart_number": "PR-10026",
        "birth_date": "1948-06-22",
        "sex": "M",
        "phone": "(71) 98877-6655",
        "email": "ricardo.mendes@email.test",
        "address": "Rua Chile 77, Salvador",
        "allergies": "none",
        "diagnosis": "Suspected sepsis, fever and hypotension.",
        "notes": "Ricardo Mendes email ricardo.mendes@email.test.",
        "exams": [
            {"name": "Lactate", "status": "pending", "result": "", "ordered_at": "2026-04-05", "resulted_at": ""},
            {"name": "Blood culture", "status": "pending", "result": "", "ordered_at": "2026-04-05", "resulted_at": ""},
        ],
    },
    {
        "name": "Patricia Gomes",
        "cpf": "404.428.201-35",
        "chart_number": "PR-10027",
        "birth_date": "1961-12-01",
        "sex": "F",
        "phone": "(85) 98765-4321",
        "email": "patricia.gomes@email.test",
        "address": "Avenida Beira Mar 500, Fortaleza",
        "allergies": "none",
        "diagnosis": "Medical inpatient, reduced mobility.",
        "notes": "Patricia Gomes, Avenida Beira Mar 500, phone (85) 98765-4321.",
        "exams": [
            {"name": "CBC", "status": "completed", "result": "Hb 11.2", "ordered_at": "2026-03-18", "resulted_at": "2026-03-18"},
        ],
    },
    {
        "name": "Luis Henrique Barros",
        "cpf": "070.680.938-68",
        "chart_number": "PR-10028",
        "birth_date": "1984-08-14",
        "sex": "M",
        "phone": "(61) 99100-2233",
        "email": "luis.barros@email.test",
        "address": "Rua 3 norte 15, Brasilia",
        "allergies": "latex",
        "diagnosis": "Upper GI bleeding, hematemesis.",
        "notes": "Luis Henrique Barros CPF 070.680.938-68, Rua 3 norte 15.",
        "exams": [
            {"name": "Hemoglobin", "status": "completed", "result": "8.4 g/dL", "ordered_at": "2026-02-11", "resulted_at": "2026-02-11"},
            {"name": "Endoscopy", "status": "pending", "result": "", "ordered_at": "2026-02-11", "resulted_at": ""},
        ],
    },
    {
        "name": "Helena Duarte",
        "cpf": "246.853.609-00",
        "chart_number": "PR-10029",
        "birth_date": "1973-05-27",
        "sex": "F",
        "phone": "(19) 98820-1010",
        "email": "helena.duarte@email.test",
        "address": "Rua Barao de Jaguara 88, Campinas",
        "allergies": "iodine contrast",
        "diagnosis": "Stable post-op day 1, VTE risk.",
        "notes": "Contrast allergy. Helena Duarte (19) 98820-1010.",
        "exams": [
            {"name": "D-dimer", "status": "pending", "result": "", "ordered_at": "2026-03-22", "resulted_at": ""},
        ],
    },
    {
        "name": "Gabriel Rocha",
        "cpf": "351.843.870-90",
        "chart_number": "PR-10030",
        "birth_date": "1998-02-02",
        "sex": "M",
        "phone": "(27) 99901-7788",
        "email": "gabriel.rocha@email.test",
        "address": "Avenida Princesa Isabel 12, Vitoria",
        "allergies": "none",
        "diagnosis": "Mild chest pain, low risk.",
        "notes": "Gabriel Rocha email gabriel.rocha@email.test.",
        "exams": [
            {"name": "Troponin", "status": "completed", "result": "negative", "ordered_at": "2026-01-09", "resulted_at": "2026-01-09"},
        ],
    },
    {
        "name": "Sofia Martins",
        "cpf": "618.442.573-01",
        "chart_number": "PR-10031",
        "birth_date": "2001-10-10",
        "sex": "F",
        "phone": "(62) 98444-5566",
        "email": "sofia.martins@email.test",
        "address": "Rua 10 200, Goiania",
        "allergies": "none",
        "diagnosis": "Pregnancy, chronic hypertension.",
        "notes": "Sofia Martins, Rua 10 200.",
        "exams": [
            {"name": "Obstetric ultrasound", "status": "completed", "result": "viable singleton", "ordered_at": "2026-02-01", "resulted_at": "2026-02-01"},
        ],
    },
    {
        "name": "Antonio Ferreira",
        "cpf": "013.276.549-17",
        "chart_number": "PR-10032",
        "birth_date": "1942-03-16",
        "sex": "M",
        "phone": "(81) 98700-1212",
        "email": "antonio.ferreira@email.test",
        "address": "Avenida Boa Viagem 1500, Recife",
        "allergies": "none",
        "diagnosis": "CABG candidate, already on atorvastatin.",
        "notes": "Antonio Ferreira CPF 013.276.549-17 Avenida Boa Viagem 1500.",
        "exams": [
            {"name": "Lipid panel", "status": "completed", "result": "LDL 78 on statin", "ordered_at": "2026-03-04", "resulted_at": "2026-03-04"},
            {"name": "Coronary angiogram", "status": "pending", "result": "", "ordered_at": "2026-03-06", "resulted_at": ""},
        ],
    },
]

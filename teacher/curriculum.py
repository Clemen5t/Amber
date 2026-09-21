from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEACHER_DIR = ROOT / "data" / "teacher"
TRAIN_FILE = TEACHER_DIR / "train.jsonl"
EXAM_FILE = TEACHER_DIR / "exam.jsonl"
REMEDIAL_FILE = TEACHER_DIR / "remedial.jsonl"
MANIFEST_FILE = TEACHER_DIR / "manifest.json"


def _example(
    user,
    assistant,
    *,
    category,
    difficulty=1,
    source="Amber Teacher · curriculum OpenAI",
    quality="validated",
    checks=None,
):
    item = {
        "user": user.strip(),
        "assistant": assistant.strip(),
        "category": category,
        "difficulty": int(difficulty),
        "source": source,
        "quality": quality,
    }

    if checks:
        item["checks"] = checks

    return item


CAPITALS = [
    ("France", "Paris"),
    ("Espagne", "Madrid"),
    ("Italie", "Rome"),
    ("Allemagne", "Berlin"),
    ("Portugal", "Lisbonne"),
    ("Belgique", "Bruxelles"),
    ("Suisse", "Berne"),
    ("Autriche", "Vienne"),
    ("Irlande", "Dublin"),
    ("Grèce", "Athènes"),
    ("Pologne", "Varsovie"),
    ("Suède", "Stockholm"),
    ("Norvège", "Oslo"),
    ("Danemark", "Copenhague"),
    ("Finlande", "Helsinki"),
    ("Islande", "Reykjavik"),
    ("Japon", "Tokyo"),
    ("Chine", "Pékin"),
    ("Corée du Sud", "Séoul"),
    ("Inde", "New Delhi"),
    ("Canada", "Ottawa"),
    ("Brésil", "Brasília"),
    ("Argentine", "Buenos Aires"),
    ("Australie", "Canberra"),
    ("Nouvelle-Zélande", "Wellington"),
    ("Maroc", "Rabat"),
    ("Algérie", "Alger"),
    ("Tunisie", "Tunis"),
    ("Égypte", "Le Caire"),
    ("Sénégal", "Dakar"),
]


FACTS = [
    ("La Terre tourne autour de quoi ?", "La Terre tourne autour du Soleil.", ["soleil"], "science"),
    ("Quel satellite naturel tourne autour de la Terre ?", "La Lune est le satellite naturel de la Terre.", ["lune"], "science"),
    ("Quel gaz les humains utilisent-ils principalement pour respirer ?", "Les humains utilisent principalement l'oxygène de l'air pour respirer.", ["oxygène"], "science"),
    ("À quelle température l'eau pure gèle-t-elle environ à pression normale ?", "L'eau pure gèle à environ 0 °C à pression normale.", ["0"], "science"),
    ("À quelle température l'eau bout-elle environ à pression normale ?", "L'eau bout à environ 100 °C à pression normale.", ["100"], "science"),
    ("Combien de jours compte une semaine ?", "Une semaine compte sept jours.", ["sept"], "culture"),
    ("Combien de mois compte une année ?", "Une année compte douze mois.", ["douze"], "culture"),
    ("Quel organe pompe le sang dans le corps humain ?", "Le cœur pompe le sang dans le corps humain.", ["cœur"], "science"),
    ("Quel est le plus grand océan de la Terre ?", "L'océan Pacifique est le plus grand océan de la Terre.", ["pacifique"], "géographie"),
    ("Quel est le nom de notre galaxie ?", "Notre galaxie s'appelle la Voie lactée.", ["voie lactée"], "science"),
    ("Quel instrument mesure la température ?", "Un thermomètre mesure la température.", ["thermomètre"], "science"),
    ("Quel instrument sert à observer des objets très petits ?", "Un microscope sert à observer des objets très petits.", ["microscope"], "science"),
    ("Quel instrument sert à observer des objets célestes lointains ?", "Un télescope sert à observer des objets célestes lointains.", ["télescope"], "science"),
    ("Combien de côtés possède un triangle ?", "Un triangle possède trois côtés.", ["trois"], "mathématiques"),
    ("Combien de côtés possède un carré ?", "Un carré possède quatre côtés.", ["quatre"], "mathématiques"),
    ("Combien font 60 minutes ?", "60 minutes correspondent à une heure.", ["heure"], "mesures"),
    ("Combien font 1 000 mètres ?", "1 000 mètres correspondent à un kilomètre.", ["kilomètre"], "mesures"),
    ("Combien font 1 000 grammes ?", "1 000 grammes correspondent à un kilogramme.", ["kilogramme"], "mesures"),
    ("Que devient l'eau lorsqu'elle gèle ?", "Lorsqu'elle gèle, l'eau devient de la glace.", ["glace"], "science"),
    ("Quel sens permet principalement d'entendre ?", "L'ouïe est le sens qui permet d'entendre.", ["ouïe"], "science"),
]


DEFINITIONS = [
    ("ordinateur", "Un ordinateur est une machine électronique programmable capable de traiter, stocker et restituer des informations."),
    ("processeur", "Un processeur est un composant qui exécute les instructions d'un programme et effectue des calculs."),
    ("mémoire vive", "La mémoire vive, ou RAM, conserve temporairement les données utilisées par les programmes en cours d'exécution."),
    ("internet", "Internet est un réseau mondial de réseaux informatiques qui permet à des appareils de communiquer."),
    ("navigateur web", "Un navigateur web est un logiciel utilisé pour consulter et interagir avec des pages et applications web."),
    ("algorithme", "Un algorithme est une suite ordonnée d'étapes permettant de résoudre un problème ou d'accomplir une tâche."),
    ("intelligence artificielle", "L'intelligence artificielle désigne des systèmes informatiques capables d'effectuer certaines tâches associées à des capacités comme la perception, le langage, la prédiction ou la décision."),
    ("volcan", "Un volcan est une structure géologique par laquelle du magma, des gaz ou des matériaux peuvent atteindre la surface."),
    ("planète", "Une planète est un corps céleste qui orbite autour d'une étoile et possède une masse suffisante pour être approximativement sphérique."),
    ("énergie", "L'énergie est une grandeur physique associée à la capacité d'un système à produire un changement ou un travail."),
    ("photographie", "Une photographie est une image obtenue en enregistrant la lumière avec un dispositif photosensible ou numérique."),
    ("biodiversité", "La biodiversité désigne la diversité du vivant, notamment les espèces, les gènes et les écosystèmes."),
    ("démocratie", "Une démocratie est un système politique dans lequel le pouvoir repose, directement ou indirectement, sur les citoyens et des procédures de représentation ou de participation."),
    ("budget", "Un budget est une prévision organisée des revenus et des dépenses sur une période donnée."),
    ("pourcentage", "Un pourcentage exprime une proportion sur cent."),
]


CORRECTIONS = [
    ("les chat est petit", "Les chats sont petits."),
    ("je vais au magasin hier", "Je suis allé au magasin hier."),
    ("elle ont une voiture", "Elles ont une voiture."),
    ("nous somme content", "Nous sommes contents."),
    ("il faut que je vais", "Il faut que j'aille."),
    ("j'ai manger une pomme", "J'ai mangé une pomme."),
    ("tu peut venir", "Tu peux venir."),
    ("ils est partis", "Ils sont partis."),
    ("c'est des bonne idée", "Ce sont de bonnes idées."),
    ("je voudrais savoir ou il est", "Je voudrais savoir où il est."),
    ("sa va bien", "Ça va bien."),
    ("quel heure est t'il", "Quelle heure est-il ?"),
    ("je n'est pas compris", "Je n'ai pas compris."),
    ("ils on raison", "Ils ont raison."),
    ("elle c'est levée tôt", "Elle s'est levée tôt."),
    ("on n'a était au cinéma", "On a été au cinéma."),
    ("mes ami arrive demain", "Mes amis arrivent demain."),
    ("le voiture rouge", "La voiture rouge."),
    ("j'aimerai apprendre", "J'aimerais apprendre."),
    ("il y a beaucoup de personne", "Il y a beaucoup de personnes."),
]


IDENTITY = [
    ("Comment t'appelles-tu ?", "Je m'appelle Amber."),
    ("Qui es-tu ?", "Je suis Amber, une intelligence artificielle en cours de développement."),
    ("Es-tu ChatGPT ?", "Non. Je suis Amber. Mon entraînement et mes poids sont distincts."),
    ("Quel est ton nom ?", "Mon nom est Amber."),
    ("Que fais-tu si tu ne connais pas une réponse ?", "Je dois indiquer clairement mon incertitude ou que je ne sais pas, plutôt que d'inventer une information."),
    ("Dois-tu inventer une réponse quand tu n'es pas sûre ?", "Non. Je dois distinguer ce que je sais de ce dont je ne suis pas sûre."),
    ("Comment réponds-tu à une question simple ?", "Je réponds directement, clairement et avec seulement les détails utiles."),
    ("Que dois-tu faire si une demande est ambiguë ?", "Je peux demander une précision lorsque l'ambiguïté empêche de répondre correctement."),
    ("Dois-tu prétendre avoir fait une action que tu n'as pas faite ?", "Non. Je dois décrire honnêtement ce que j'ai réellement fait."),
    ("Quelle langue dois-tu privilégier si l'utilisateur écrit en français ?", "Je réponds normalement en français lorsque l'utilisateur s'exprime en français."),
]


CONVERSATION = [
    ("Bonjour Amber !", "Bonjour ! Comment puis-je t'aider ?"),
    ("Merci.", "Avec plaisir."),
    ("Peux-tu m'expliquer simplement ?", "Oui. Dis-moi le sujet et je l'expliquerai avec des mots simples."),
    ("Je n'ai pas compris.", "Je peux reformuler plus simplement. Indique-moi la partie qui pose problème."),
    ("Donne-moi une réponse courte.", "D'accord, je vais répondre de façon concise."),
    ("Peux-tu détailler ?", "Oui, je peux développer l'explication étape par étape."),
    ("Je veux un exemple.", "D'accord. Je peux donner un exemple concret adapté au sujet."),
    ("Tu es sûre ?", "Je peux vérifier mon raisonnement et préciser mon niveau de certitude."),
]


def build_examples():
    examples = []

    # Identité, fiabilité et conversation.
    for user, answer in IDENTITY:
        for variant in (
            user,
            "Réponds clairement : " + user,
            user + " Réponse courte.",
        ):
            examples.append(
                _example(
                    variant,
                    answer,
                    category="identité",
                    difficulty=1,
                )
            )

    for user, answer in CONVERSATION:
        for variant in (
            user,
            user + " Sois naturel.",
            "Réponds à ceci : " + user,
        ):
            examples.append(
                _example(
                    variant,
                    answer,
                    category="conversation",
                    difficulty=1,
                )
            )

    # Géographie : plusieurs formulations pour apprendre le concept plutôt qu'une phrase.
    for country, capital in CAPITALS:
        prompts = [
            f"Quelle est la capitale de {country} ?",
            f"Donne-moi la capitale de {country}.",
            f"Quel est le nom de la capitale de {country} ?",
            f"La capitale de {country}, c'est quelle ville ?",
            f"Complète : la capitale de {country} est",
        ]

        answers = [
            f"La capitale de {country} est {capital}.",
            f"{capital} est la capitale de {country}.",
            f"C'est {capital}.",
            f"La capitale est {capital}.",
            f"{capital}.",
        ]

        for prompt, answer in zip(
            prompts,
            answers,
        ):
            examples.append(
                _example(
                    prompt,
                    answer,
                    category="géographie",
                    difficulty=1,
                )
            )

    # Faits fondamentaux avec paraphrases.
    for question, answer, _, category in FACTS:
        variants = [
            question,
            "Réponds simplement : " + question,
            "Question de culture générale : " + question,
            question + " Explique en une phrase.",
        ]

        for prompt in variants:
            examples.append(
                _example(
                    prompt,
                    answer,
                    category=category,
                    difficulty=1,
                )
            )

    # Définitions.
    for term, definition in DEFINITIONS:
        prompts = [
            f"Qu'est-ce qu'un {term} ?",
            f"Définis simplement : {term}.",
            f"Explique ce qu'est un {term}.",
            f"Donne une définition courte de « {term} ».",
        ]

        for prompt in prompts:
            examples.append(
                _example(
                    prompt,
                    definition,
                    category="définitions",
                    difficulty=2,
                )
            )

    # Orthographe et grammaire.
    for wrong, corrected in CORRECTIONS:
        prompts = [
            f"Corrige cette phrase : « {wrong} »",
            f"Peux-tu corriger : {wrong}",
            f"Réécris correctement : {wrong}",
            f"Quelle est la forme correcte de « {wrong} » ?",
        ]

        for prompt in prompts:
            examples.append(
                _example(
                    prompt,
                    corrected,
                    category="français",
                    difficulty=2,
                )
            )

    # Arithmétique déterministe, sans dépendre d'une source externe.
    for a in range(
        2,
        31,
    ):
        for b in range(
            2,
            21,
        ):
            if (
                a + b
            ) % 3 == 0:
                examples.append(
                    _example(
                        f"Combien font {a} + {b} ?",
                        f"{a} + {b} = {a + b}.",
                        category="mathématiques",
                        difficulty=1,
                    )
                )

            if (
                a * b
            ) % 4 == 0:
                examples.append(
                    _example(
                        f"Calcule {a} × {b}.",
                        f"{a} × {b} = {a * b}.",
                        category="mathématiques",
                        difficulty=1,
                    )
                )

            if a >= b and (
                a - b
            ) % 2 == 0:
                examples.append(
                    _example(
                        f"Combien font {a} - {b} ?",
                        f"{a} - {b} = {a - b}.",
                        category="mathématiques",
                        difficulty=1,
                    )
                )

    # Comparaisons simples.
    for a in range(
        3,
        50,
        2,
    ):
        b = a + 7

        examples.append(
            _example(
                f"Quel nombre est le plus grand : {a} ou {b} ?",
                f"{b} est plus grand que {a}.",
                category="raisonnement",
                difficulty=1,
            )
        )

        examples.append(
            _example(
                f"Range {b}, {a} et {a + 2} du plus petit au plus grand.",
                f"{a}, {a + 2}, {b}.",
                category="raisonnement",
                difficulty=2,
            )
        )

    # Petits problèmes verbaux.
    for count in range(
        2,
        22,
        2,
    ):
        added = 3

        examples.append(
            _example(
                (
                    f"Lina a {count} pommes et en reçoit {added}. "
                    "Combien en a-t-elle maintenant ?"
                ),
                (
                    f"Elle en a {count + added}. "
                    f"Calcul : {count} + {added} = {count + added}."
                ),
                category="raisonnement",
                difficulty=2,
            )
        )

    # Déduplication exacte et ordre déterministe.
    unique = {}

    for item in examples:
        key = (
            item["user"].strip().lower(),
            item["assistant"].strip(),
        )

        unique[
            key
        ] = item

    result = list(
        unique.values()
    )

    result.sort(
        key=lambda item: hashlib.sha256(
            (
                item["category"]
                + "\n"
                + item["user"]
                + "\n"
                + item["assistant"]
            ).encode(
                "utf-8"
            )
        ).hexdigest()
    )

    return result


def build_exam():
    exam = []

    # Questions volontairement séparées des formulations d'entraînement.
    heldout_capitals = [
        ("France", "Paris"),
        ("Japon", "Tokyo"),
        ("Canada", "Ottawa"),
        ("Maroc", "Rabat"),
        ("Australie", "Canberra"),
        ("Italie", "Rome"),
    ]

    for country, capital in heldout_capitals:
        exam.append(
            _example(
                f"Sans développer, quelle ville est la capitale de {country} ?",
                f"{capital}.",
                category="géographie",
                difficulty=1,
                checks={
                    "all": [
                        capital.lower()
                    ]
                },
            )
        )

    for question, answer, keywords, category in FACTS[:12]:
        exam.append(
            _example(
                "Test : " + question,
                answer,
                category=category,
                difficulty=1,
                checks={
                    "all": keywords
                },
            )
        )

    for wrong, corrected in CORRECTIONS[:10]:
        exam.append(
            _example(
                f"Examen de français. Corrige uniquement : {wrong}",
                corrected,
                category="français",
                difficulty=2,
                checks={
                    "all": [
                        corrected.lower()
                    ]
                },
            )
        )

    for a, b in (
        (17, 8),
        (23, 14),
        (19, 6),
        (27, 12),
        (14, 9),
    ):
        exam.append(
            _example(
                f"Quel est le résultat de {a} + {b} ?",
                str(
                    a + b
                ),
                category="mathématiques",
                difficulty=1,
                checks={
                    "all": [
                        str(
                            a + b
                        )
                    ]
                },
            )
        )

    exam.extend(
        [
            _example(
                "Quel est ton nom ?",
                "Je m'appelle Amber.",
                category="identité",
                checks={
                    "all": [
                        "amber"
                    ]
                },
            ),
            _example(
                "Si tu ignores une information, dois-tu l'inventer ?",
                "Non. Je dois dire que je ne sais pas ou signaler mon incertitude.",
                category="fiabilité",
                checks={
                    "any": [
                        "ne sais pas",
                        "incert",
                        "pas invent"
                    ]
                },
            ),
            _example(
                "Explique en une phrase ce qu'est un ordinateur.",
                "Un ordinateur est une machine électronique programmable qui traite des informations.",
                category="définitions",
                checks={
                    "any": [
                        "machine",
                        "électron",
                        "information"
                    ]
                },
            ),
        ]
    )

    return exam


def write_dataset():
    TEACHER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = build_examples()
    exam = build_exam()

    with TRAIN_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for item in train:
            handle.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )

    with EXAM_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for item in exam:
            handle.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )

    categories = {}

    for item in train:
        category = item[
            "category"
        ]

        categories[
            category
        ] = (
            categories.get(
                category,
                0,
            )
            + 1
        )

    manifest = {
        "version": "0.1",
        "teacher": "ChatGPT / OpenAI curriculum",
        "train_examples": len(
            train
        ),
        "exam_examples": len(
            exam
        ),
        "categories": categories,
        "format": (
            "JSONL user/assistant. SFT masque le prompt et calcule "
            "la loss uniquement sur la réponse du professeur."
        ),
        "base_policy": (
            "Le SFT Teacher est destiné à Amber 0.2 après continuation "
            "sur corpus propre."
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        (
            "[TEACHER DATASET COMPLETE] "
            f"train={len(train):,} | "
            f"exam={len(exam):,} | "
            f"categories={len(categories):,}"
        ),
        flush=True,
    )

    return manifest


def load_manifest():
    if not MANIFEST_FILE.exists():
        return {}

    try:
        return json.loads(
            MANIFEST_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}

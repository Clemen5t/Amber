import unittest

from amber.dataset_v02 import clean_document


class AmberV02CleaningTests(unittest.TestCase):

    def test_natural_french_prose_is_kept(self):
        text = (
            "Paris\n"
            "Paris est la capitale de la France et une grande ville européenne. "
            "Elle se situe sur les rives de la Seine et possède une histoire "
            "riche, de nombreux musées, des universités et plusieurs monuments "
            "connus. La ville joue aussi un rôle important dans la culture, "
            "la recherche, l'économie et les transports du pays."
        )

        cleaned, reason = clean_document(
            text
        )

        self.assertEqual(
            reason,
            "ok"
        )

        self.assertIsNotNone(
            cleaned
        )

        self.assertIn(
            "capitale de la France",
            cleaned
        )

    def test_wiki_source_noise_is_removed_or_rejected(self):
        text = (
            "Exemple\n"
            "| source=, Le roman d'un enfant, page 288\n"
            "{{exemple|lang=fr}}\n"
            "* :\n"
            "* :\n"
            "* :\n"
            "#*\n"
            "|-\n"
            "| style=width:20%\n"
        )

        cleaned, reason = clean_document(
            text
        )

        self.assertIsNone(
            cleaned
        )

        self.assertNotEqual(
            reason,
            "ok"
        )

    def test_repeated_symbol_lines_do_not_survive(self):
        text = (
            "Animal\n"
            "Le chat est un mammifère domestique présent dans de nombreux "
            "foyers. Il communique par différents sons et comportements, "
            "possède une bonne vision nocturne et peut vivre au contact des "
            "humains. Les chats ont été associés à de nombreuses cultures "
            "et occupent aujourd'hui une place importante parmi les animaux "
            "de compagnie.\n"
            "* :\n"
            "* :\n"
            "* :\n"
            "* :\n"
        )

        cleaned, reason = clean_document(
            text
        )

        self.assertEqual(
            reason,
            "ok"
        )

        self.assertNotIn(
            "* :",
            cleaned
        )


if __name__ == "__main__":
    unittest.main()

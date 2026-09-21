import unittest

from teacher.curriculum import (
    build_examples,
    build_exam,
)


class AmberTeacherCurriculumTests(
    unittest.TestCase
):

    def test_curriculum_has_substantial_variety(self):
        examples = build_examples()

        self.assertGreaterEqual(
            len(
                examples
            ),
            700,
        )

        categories = {
            item[
                "category"
            ]
            for item in examples
        }

        self.assertGreaterEqual(
            len(
                categories
            ),
            7,
        )

    def test_examples_are_unique(self):
        examples = build_examples()

        keys = [
            (
                item[
                    "user"
                ].strip().lower(),
                item[
                    "assistant"
                ].strip(),
            )
            for item in examples
        ]

        self.assertEqual(
            len(
                keys
            ),
            len(
                set(
                    keys
                )
            ),
        )

    def test_exam_is_held_out_format_and_scored(self):
        exam = build_exam()

        self.assertGreaterEqual(
            len(
                exam
            ),
            30,
        )

        for item in exam:
            self.assertTrue(
                item.get(
                    "checks"
                )
            )


if __name__ == "__main__":
    unittest.main()

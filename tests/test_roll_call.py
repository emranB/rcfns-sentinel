'''Check that roll calls produce only well-supported membership counts.'''

import unittest

from src.backend.roll_call import RollCallCount, roll_call_count

SOURCE = "Planning Committee/Meeting Minutes.docx"
QUESTION = "How many planning committee members do we have?"
ROLL_CALL = (
    "Present: Mei Tan, Amélie D'Arcy, A. Jean-Luc van Dijk\n"
    "Regrets: Noor Khan; Priya Patel\n"
    "Staff: Jordan Gray\n"
    "Guest: Lee Green"
)


class RollCallTests(unittest.TestCase):
    # Count both member sections and exclude staff and guests.
    def test_counts_members_in_a_named_group(self) -> None:
        result = roll_call_count(QUESTION, ROLL_CALL, source=SOURCE)
        self.assertEqual(result, RollCallCount(present=3, absent=2))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.total, 5)

    # Allow an explicit empty absence list when both labels specify members.
    def test_counts_distinct_members_with_explicit_membership_labels(self) -> None:
        result = roll_call_count(
            "How many members are on the planning committee today?",
            "Members present: Amélie D’Arcy, Amélie D'Arcy\nMembers absent: None",
            source=SOURCE,
        )
        self.assertEqual(result, RollCallCount(present=1, absent=0))

    # Leave attendance, dated, mixed, and wrong-group questions to other evidence.
    def test_declines_questions_that_are_not_simple_membership_totals(self) -> None:
        questions = (
            "How many planning committee members were present?",
            "How many planning committee members do we have, and what are their names?",
            "How many planning committee members did we have in 2024?",
            "How many staff members do we have?",
            "How many board members do we have?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNone(roll_call_count(question, ROLL_CALL, source=SOURCE))

    # Refuse partial, cross-section, contradictory, or non-name material.
    def test_declines_uncertain_roll_calls(self) -> None:
        excerpts = (
            "Present: Mei Tan\nRegrets: Priya Patel",
            "Present: Mei Tan\nNew business\nRegrets: Priya Patel\nGuest: Lee Green",
            "Present: Mei Tan\nRegrets: Mei Tan\nStaff: Jordan Gray",
            "Present: Draft Minutes\nRegrets: Priya Patel\nStaff: Jordan Gray",
            "Present: Mei Tan (proxy for a member)\nRegrets: Priya Patel\nStaff: Jordan Gray",
        )
        for excerpt in excerpts:
            with self.subTest(excerpt=excerpt):
                self.assertIsNone(roll_call_count(QUESTION, excerpt, source=SOURCE))

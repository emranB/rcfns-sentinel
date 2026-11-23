"""Check that roll calls produce only well-supported membership counts."""

import unittest

from src.backend.roll_call import (
    RollCallCount,
    contact_directory,
    dated_count_scope,
    dated_tenure_names,
    named_attendance_section,
    parse_roll_call,
    roll_call_count,
    roster_aliases,
    roster_group,
    roster_names,
)

SOURCE = "Planning Committee/Meeting Minutes.docx"
QUESTION = "How many planning committee members do we have?"
ROLL_CALL = (
    "Present: Mei Tan, Amélie D'Arcy, A. Jean-Luc van Dijk\n"
    "Regrets: Noor Khan; Priya Patel\n"
    "Staff: Jordan Gray\n"
    "Guest: Lee Green"
)


class RollCallTests(unittest.TestCase):
    # Open-ended, overlapping and employment rows cannot imply membership in later years.
    def test_historical_register_requires_dated_service(self) -> None:
        rows = [
            ["Member name", "Date appointed", "Year retired"],
            ["Mei Tan", "2010", "2014 (April)"],
            ["Noor Khan", "2014", "Current"],
            ["Ari King", "2011", "Current"],
            ["Jordan Gray", "Hired - 2014", "2018"],
            ["Sam Green", "2014", "2013"],
        ]
        self.assertEqual(dated_tenure_names([rows], 2014), ("Mei Tan", "Noor Khan"))
        self.assertEqual(dated_tenure_names([rows], 2018), ())

    # Extract membership from a declared directory without returning support contacts.
    def test_contact_directory_keeps_members_and_stops_at_support(self) -> None:
        result = contact_directory(
            "2016 Library Council Membership List\n"
            "Mei Tan, Vice Chair\tSuite 3\n"
            "Noor Khan\tSuite 4\n"
            "Council Support\nJordan Gray\tSuite 5"
        )
        self.assertEqual(
            result, ("2016 Library Council Membership List", ("Mei Tan", "Noor Khan"))
        )

    # Document section labels apply only when a single unnegated status is named.
    def test_named_attendance_sections_are_unambiguous(self) -> None:
        self.assertEqual(named_attendance_section("Who sent apologies?"), "absent")
        self.assertEqual(
            named_attendance_section("Names listed as present?"), "present"
        )
        self.assertIsNone(named_attendance_section("Who wasn't absent?"))
        self.assertIsNone(named_attendance_section("Present the full roster."))
        self.assertIsNone(
            named_attendance_section("Who was present and who sent regrets?")
        )

    # Keep original spelling and attendance status for subsequent roster requests.
    def test_count_and_roster_share_the_same_entries(self) -> None:
        call = parse_roll_call(QUESTION, ROLL_CALL, source=SOURCE)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(
            call.present, ("Mei Tan", "Amélie D'Arcy", "A. Jean-Luc van Dijk")
        )
        self.assertEqual(call.absent, ("Noor Khan", "Priya Patel"))
        self.assertEqual(call.count.total, 5)

    # Resolve short headcount questions without treating attendance as membership.
    def test_dated_counts_identify_group_and_year(self) -> None:
        self.assertEqual(
            dated_count_scope("How many council members in 2024?"), ("council", 2024)
        )
        self.assertEqual(
            dated_count_scope("How many directors in 2023?"), ("board", 2023)
        )
        self.assertEqual(
            dated_count_scope("How many trustees were on the library council in 2022?"),
            ("library council", 2022),
        )
        self.assertIsNone(
            dated_count_scope("How many council members were present in 2024?")
        )
        self.assertIsNone(dated_count_scope("How many directors in 2023 and 2024?"))

    # Accept a single historical period and a misspelled membership role.
    def test_roster_group_accepts_one_year_and_a_misspelled_role(self) -> None:
        self.assertEqual(roster_group("List all council memeberrs for 2024"), "council")
        self.assertEqual(roster_group("List all council expenses for 2024"), "")

    # Recover wrapped names and preserve the distinction between members and attendees.
    def test_roster_includes_regrets_and_excludes_attendees_with_other_roles(
        self,
    ) -> None:
        excerpt = (
            "Meeting minutes\nAttendees\nAlex Wood - ED, Reba\n"
            "Torres Mendez - Trustee, Jamie Du - Chair\nRegrets\nEllis Smith - Trustee\n"
            "Agenda approval\nChris Banks - Trustee\n"
        )
        self.assertEqual(
            roster_names(excerpt, "planning committee", SOURCE),
            ["Reba Torres Mendez", "Jamie Du", "Ellis Smith"],
        )
        self.assertEqual(roster_names(excerpt, "finance committee", SOURCE), [])

    # Preserve separately listed people even when their recorded names are similar.
    def test_roster_aliases_require_compatible_roll_calls(self) -> None:
        names = roster_aliases(
            [
                ["Leona Martinez", "Ravi Shah", "Antonio Del Toro"],
                ["Leona Martnez", "Ravi Shah", "Ravi Shah Jr", "Antonio Del Toro"],
                ["Leona Martinez", "Antonio Del"],
            ]
        )
        self.assertEqual(names["Leona Martnez"], "Leona Martinez")
        self.assertEqual(names["Antonio Del"], "Antonio Del Toro")
        self.assertEqual(names["Ravi Shah"], "Ravi Shah")

    # Count both member sections and exclude staff and guests.
    def test_counts_members_in_a_named_group(self) -> None:
        result = roll_call_count(QUESTION, ROLL_CALL, source=SOURCE)
        self.assertEqual(result, RollCallCount(present=3, absent=2))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.total, 5)

    # Count directors whose names include an explicit role in the roll call.
    def test_counts_members_with_role_labels(self) -> None:
        result = roll_call_count(
            QUESTION,
            "Present: Mei Tan (Chair), Priya Patel (Trustee)\nRegrets: Noor Khan\nStaff: Jordan Gray",
            source=SOURCE,
        )
        self.assertEqual(result, RollCallCount(present=2, absent=1))

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

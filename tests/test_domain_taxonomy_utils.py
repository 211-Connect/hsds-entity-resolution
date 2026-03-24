"""Tests for domain_utils and taxonomy_utils.

Tests specify correct HSDS entity resolution behavior.
"""

from __future__ import annotations

from hsds_entity_resolution.core.domain_utils import (
    domain_overlap_score,
    extract_contact_domains,
    extract_domain,
)
from hsds_entity_resolution.core.taxonomy_utils import (
    clean_services_rollup,
    clean_taxonomy_objects,
    extract_entity_taxonomy_codes,
    extract_taxonomy_codes,
    taxonomy_parent_codes,
)

# ===========================================================================
# domain_utils — extract_domain
# ===========================================================================


class TestExtractDomain:
    """Unit tests for the extract_domain function."""

    def test_plain_email_returns_domain(self) -> None:
        assert extract_domain("contact@northshelter.org") == "northshelter.org"

    def test_email_with_mailto_prefix_strips_prefix(self) -> None:
        assert extract_domain("mailto:contact@northshelter.org") == "northshelter.org"

    def test_email_uppercase_normalized_to_lowercase(self) -> None:
        assert extract_domain("User@NorthShelter.ORG") == "northshelter.org"

    def test_plain_website_returns_domain(self) -> None:
        assert extract_domain("northshelter.org") == "northshelter.org"

    def test_website_with_https_scheme_strips_scheme(self) -> None:
        assert extract_domain("https://northshelter.org") == "northshelter.org"

    def test_website_with_http_scheme_strips_scheme(self) -> None:
        assert extract_domain("http://northshelter.org") == "northshelter.org"

    def test_website_www_prefix_stripped(self) -> None:
        assert extract_domain("www.northshelter.org") == "northshelter.org"

    def test_website_www_with_scheme_stripped(self) -> None:
        assert extract_domain("https://www.northshelter.org") == "northshelter.org"

    def test_website_with_port_strips_port(self) -> None:
        assert extract_domain("northshelter.org:8080") == "northshelter.org"

    def test_website_with_path_strips_path(self) -> None:
        assert extract_domain("northshelter.org/about/contact") == "northshelter.org"

    def test_website_with_query_string_strips_query(self) -> None:
        assert extract_domain("northshelter.org?utm_source=test") == "northshelter.org"

    def test_website_with_auth_credentials_strips_credentials(self) -> None:
        assert extract_domain("user:pass@northshelter.org") == "northshelter.org"

    def test_none_returns_none(self) -> None:
        assert extract_domain(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_domain("") is None

    def test_integer_returns_none(self) -> None:
        assert extract_domain(42) is None  # type: ignore[arg-type]

    def test_whitespace_only_returns_none(self) -> None:
        assert extract_domain("   ") is None

    def test_email_missing_local_part_returns_none(self) -> None:
        """An email with no local part before '@' is invalid and should return None."""
        assert extract_domain("@northshelter.org") is None

    def test_email_missing_domain_part_returns_none(self) -> None:
        """An email with no domain after '@' is invalid and should return None."""
        assert extract_domain("contact@") is None


# ===========================================================================
# domain_utils — extract_contact_domains (email + website deduplication)
# ===========================================================================


class TestExtractContactDomains:
    """Unit tests for extract_contact_domains (email + website union)."""

    def test_single_email_returns_domain(self) -> None:
        result = extract_contact_domains(
            emails_value=["contact@northshelter.org"],
            websites_value=[],
        )
        assert result == {"northshelter.org"}

    def test_single_website_returns_domain(self) -> None:
        result = extract_contact_domains(
            emails_value=[],
            websites_value=["northshelter.org"],
        )
        assert result == {"northshelter.org"}

    def test_email_and_website_with_same_root_are_deduplicated(self) -> None:
        """Tier B: a shared domain from email + website counts once, not twice.

        Domain-overlap scoring divides by the size of the union set, so
        double-counting a domain would artificially inflate the score.
        """
        result = extract_contact_domains(
            emails_value=["contact@northshelter.org"],
            websites_value=["northshelter.org"],
        )
        assert result == {"northshelter.org"}
        assert len(result) == 1

    def test_multiple_different_domains_all_returned(self) -> None:
        result = extract_contact_domains(
            emails_value=["alice@acme.org", "bob@widgets.org"],
            websites_value=[],
        )
        assert result == {"acme.org", "widgets.org"}

    def test_empty_inputs_return_empty_set(self) -> None:
        result = extract_contact_domains(emails_value=[], websites_value=[])
        assert result == set()

    def test_none_inputs_return_empty_set(self) -> None:
        result = extract_contact_domains(emails_value=None, websites_value=None)
        assert result == set()

    def test_subdomains_preserved_as_distinct_entries(self) -> None:
        """extract_contact_domains always returns full hostnames, never collapses subdomains.

        Subdomain collapsing and graded overlap are responsibilities of
        domain_overlap_score, not extract_contact_domains.  This keeps the two
        concerns cleanly separated.
        """
        result = extract_contact_domains(
            emails_value=["info@services.unitedway.org"],
            websites_value=["outreach.unitedway.org"],
        )
        assert "services.unitedway.org" in result
        assert "outreach.unitedway.org" in result
        assert len(result) == 2


# ===========================================================================
# domain_utils — domain_overlap_score (graded URL matching)
# ===========================================================================


class TestDomainOverlapScore:
    """Graded overlap scoring via domain_overlap_score.

    Score tiers:
        1.0 — same hostname, no path on either side
        0.8 — same hostname, path present on at least one URL
        0.4 — same eTLD+1, different subdomains
        0.0 — no overlap or empty inputs
    """

    # -------------------------------------------------------------------
    # 1.0 — clean root match (same host, no path on either side)
    # -------------------------------------------------------------------

    def test_identical_bare_domains_score_1(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=[],
                left_websites=["https://www.abc.com"],
                right_emails=[],
                right_websites=["https://www.abc.com"],
            )
            == 1.0
        )

    def test_www_vs_bare_same_host_score_1(self) -> None:
        """www. prefix is stripped; www.abc.com and abc.com are the same host."""
        assert (
            domain_overlap_score(
                left_emails=[],
                left_websites=["https://www.abc.com"],
                right_emails=[],
                right_websites=["https://abc.com"],
            )
            == 1.0
        )

    def test_http_vs_https_same_host_score_1(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=[],
                left_websites=["http://abc.com"],
                right_emails=[],
                right_websites=["https://abc.com"],
            )
            == 1.0
        )

    def test_email_domain_matches_website_root_score_1(self) -> None:
        """An email whose domain equals the website root is a clean root match."""
        assert (
            domain_overlap_score(
                left_emails=["contact@northshelter.org"],
                left_websites=[],
                right_emails=[],
                right_websites=["https://northshelter.org"],
            )
            == 1.0
        )

    # -------------------------------------------------------------------
    # 0.8 — same host but at least one URL carries a path slug
    # -------------------------------------------------------------------

    def test_website_with_path_vs_root_score_08(self) -> None:
        """One side has a path slug; same hostname → 0.8."""
        score = domain_overlap_score(
            left_emails=[],
            left_websites=["https://www.abc.com/programs"],
            right_emails=[],
            right_websites=["https://abc.com"],
        )
        assert score == 0.8

    def test_both_websites_with_different_paths_score_08(self) -> None:
        score = domain_overlap_score(
            left_emails=[],
            left_websites=["https://abc.com/services"],
            right_emails=[],
            right_websites=["https://abc.com/contact"],
        )
        assert score == 0.8

    def test_both_websites_with_same_path_score_08(self) -> None:
        score = domain_overlap_score(
            left_emails=[],
            left_websites=["https://abc.com/services"],
            right_emails=[],
            right_websites=["https://abc.com/services"],
        )
        assert score == 0.8

    # -------------------------------------------------------------------
    # 0.4 — same eTLD+1, different subdomains (umbrella-org signal)
    # -------------------------------------------------------------------

    def test_different_subdomains_same_root_score_04(self) -> None:
        """services.unitedway.org vs outreach.unitedway.org: same eTLD+1, different sub."""
        score = domain_overlap_score(
            left_emails=[],
            left_websites=["services.unitedway.org"],
            right_emails=[],
            right_websites=["outreach.unitedway.org"],
        )
        assert score == 0.4

    def test_numbered_subdomain_vs_www_same_root_score_04(self) -> None:
        """https://123.abc.com vs https://www.abc.com: same registered domain."""
        score = domain_overlap_score(
            left_emails=[],
            left_websites=["https://123.abc.com"],
            right_emails=[],
            right_websites=["https://www.abc.com"],
        )
        assert score == 0.4

    def test_email_subdomain_vs_different_website_subdomain_score_04(self) -> None:
        score = domain_overlap_score(
            left_emails=["info@services.unitedway.org"],
            left_websites=[],
            right_emails=[],
            right_websites=["outreach.unitedway.org"],
        )
        assert score == 0.4

    def test_best_score_wins_across_multiple_contacts(self) -> None:
        """If one contact pair gives 1.0 and another gives 0.4, the score is 1.0."""
        score = domain_overlap_score(
            left_emails=["contact@unitedway.org"],
            left_websites=["services.unitedway.org"],
            right_emails=[],
            right_websites=["unitedway.org"],
        )
        assert score == 1.0

    # -------------------------------------------------------------------
    # 0.0 — no match or empty inputs
    # -------------------------------------------------------------------

    def test_completely_different_domains_score_0(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=[],
                left_websites=["https://acme.org"],
                right_emails=[],
                right_websites=["https://widgets.com"],
            )
            == 0.0
        )

    def test_empty_left_score_0(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=[],
                left_websites=[],
                right_emails=[],
                right_websites=["https://abc.com"],
            )
            == 0.0
        )

    def test_empty_right_score_0(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=["contact@abc.com"],
                left_websites=[],
                right_emails=[],
                right_websites=[],
            )
            == 0.0
        )

    def test_none_inputs_score_0(self) -> None:
        assert (
            domain_overlap_score(
                left_emails=None,
                left_websites=None,
                right_emails=None,
                right_websites=None,
            )
            == 0.0
        )


# ===========================================================================
# taxonomy_utils — clean_taxonomy_objects
# ===========================================================================


class TestCleanTaxonomyObjects:
    """Tests for taxonomy object normalization and deduplication."""

    def test_string_items_converted_to_code_objects(self) -> None:
        result = clean_taxonomy_objects(["BD-1800", "YF-3000"])
        codes = [item["code"] for item in result]
        assert "bd-1800" in codes
        assert "yf-3000" in codes

    def test_dict_items_with_code_key_normalized(self) -> None:
        result = clean_taxonomy_objects([{"code": "BD-1800", "name": "Health Services"}])
        assert len(result) == 1
        assert result[0]["code"] == "bd-1800"
        assert result[0]["name"] == "Health Services"

    def test_uppercase_CODE_alias_key_recognized(self) -> None:
        result = clean_taxonomy_objects([{"CODE": "BD-1800"}])
        assert len(result) == 1
        assert result[0]["code"] == "bd-1800"

    def test_taxonomy_code_alias_key_recognized(self) -> None:
        result = clean_taxonomy_objects([{"taxonomy_code": "BD-1800"}])
        assert len(result) == 1
        assert result[0]["code"] == "bd-1800"

    def test_camelcase_taxonomyCode_alias_key_recognized(self) -> None:
        result = clean_taxonomy_objects([{"taxonomyCode": "BD-1800"}])
        assert len(result) == 1
        assert result[0]["code"] == "bd-1800"

    def test_taxonomy_term_id_preserved_when_present(self) -> None:
        """Canonical taxonomy payloads must retain taxonomy_term_id for persistence."""
        result = clean_taxonomy_objects(
            [{"taxonomy_term_id": "tax-1", "code": "BD-1800", "name": "Health Services"}]
        )
        assert len(result) == 1
        assert result[0]["taxonomy_term_id"] == "tax-1"
        assert result[0]["code"] == "bd-1800"

    def test_duplicate_codes_deduplicated_first_in_wins(self) -> None:
        """When two items resolve to the same code, the first one is kept."""
        result = clean_taxonomy_objects(
            [
                {"code": "BD-1800", "name": "First Occurrence"},
                {"taxonomy_code": "BD-1800", "name": "Second Occurrence"},
            ]
        )
        assert len(result) == 1
        assert result[0]["name"] == "First Occurrence"

    def test_mixed_case_duplicates_deduplicated(self) -> None:
        """'BD-1800' and 'bd-1800' resolve to the same code and deduplicate."""
        result = clean_taxonomy_objects(["BD-1800", "bd-1800"])
        assert len(result) == 1

    def test_output_sorted_by_code_for_deterministic_hashing(self) -> None:
        """Output must be sorted by code value so that hash input is stable."""
        result = clean_taxonomy_objects(["ZZ-9999", "AA-0001", "MM-5555"])
        codes = [str(item["code"]) for item in result]
        assert codes == sorted(codes)

    def test_empty_list_returns_empty(self) -> None:
        assert clean_taxonomy_objects([]) == []

    def test_non_list_input_returns_empty(self) -> None:
        assert clean_taxonomy_objects(None) == []  # type: ignore[arg-type]
        assert clean_taxonomy_objects("not-a-list") == []  # type: ignore[arg-type]

    def test_dict_without_known_alias_key_skipped(self) -> None:
        """A dict with no recognized code key must be silently skipped."""
        result = clean_taxonomy_objects([{"unknown_key": "BD-1800"}])
        assert result == []


# ===========================================================================
# taxonomy_utils — taxonomy_parent_codes
# ===========================================================================


class TestTaxonomyParentCodes:
    """Tests for hierarchical parent code expansion."""

    def test_three_level_code_expands_to_two_parents(self) -> None:
        """'a-b-c' must expand to {'a', 'a-b'} — upward only."""
        result = taxonomy_parent_codes("a-b-c")
        assert result == {"a", "a-b"}

    def test_realistic_hsds_code_expands_correctly(self) -> None:
        """'bd-1800.2000-620' must expand to 'bd' and 'bd-1800.2000'."""
        result = taxonomy_parent_codes("bd-1800.2000-620")
        assert "bd" in result
        assert "bd-1800.2000" in result
        assert "bd-1800.2000-620" not in result

    def test_two_level_code_expands_to_one_parent(self) -> None:
        assert taxonomy_parent_codes("bd-1800") == {"bd"}

    def test_single_level_code_returns_empty_set(self) -> None:
        """A code with no dashes has no parent codes."""
        assert taxonomy_parent_codes("bd") == set()

    def test_expansion_is_upward_only(self) -> None:
        """Parent expansion must go upward (more general), never downward."""
        result = taxonomy_parent_codes("bd-1800")
        # 'bd-1800-100' (child) must NOT appear
        assert not any("-" in code and code.startswith("bd-1800-") for code in result)

    def test_empty_string_returns_empty_set(self) -> None:
        assert taxonomy_parent_codes("") == set()


# ===========================================================================
# taxonomy_utils — extract_entity_taxonomy_codes
# ===========================================================================


class TestExtractEntityTaxonomyCodes:
    """Tests for combined taxonomy + service code extraction."""

    def test_codes_extracted_from_taxonomies_field(self) -> None:
        entity = {"taxonomies": [{"code": "BD-1800"}]}
        result = extract_entity_taxonomy_codes(entity=entity)
        assert "bd-1800" in result

    def test_codes_extracted_from_services_rollup(self) -> None:
        entity = {
            "taxonomies": [],
            "services_rollup": [{"name": "Service A", "taxonomies": [{"code": "YF-3000"}]}],
        }
        result = extract_entity_taxonomy_codes(entity=entity)
        assert "yf-3000" in result

    def test_codes_from_both_sources_merged(self) -> None:
        entity = {
            "taxonomies": [{"code": "BD-1800"}],
            "services_rollup": [{"name": "Service A", "taxonomies": [{"code": "YF-3000"}]}],
        }
        result = extract_entity_taxonomy_codes(entity=entity)
        assert "bd-1800" in result
        assert "yf-3000" in result

    def test_include_parent_codes_expands_all_codes(self) -> None:
        """With include_parent_codes=True, parent codes of ALL collected codes are added."""
        entity = {
            "taxonomies": [{"code": "BD-1800"}],
            "services_rollup": [{"name": "S", "taxonomies": [{"code": "YF-3000"}]}],
        }
        result = extract_entity_taxonomy_codes(entity=entity, include_parent_codes=True)
        assert "bd" in result
        assert "yf" in result

    def test_without_parent_codes_parents_not_included(self) -> None:
        entity = {"taxonomies": [{"code": "BD-1800"}]}
        result = extract_entity_taxonomy_codes(entity=entity, include_parent_codes=False)
        assert "bd" not in result

    def test_uppercase_key_aliases_recognized(self) -> None:
        """TAXONOMIES (uppercase) must be recognized as an alias for taxonomies."""
        entity = {"TAXONOMIES": [{"code": "BD-1800"}]}
        result = extract_entity_taxonomy_codes(entity=entity)
        assert "bd-1800" in result

    def test_empty_entity_returns_empty_set(self) -> None:
        assert extract_entity_taxonomy_codes(entity={}) == set()


# ===========================================================================
# taxonomy_utils — extract_taxonomy_codes (standalone)
# ===========================================================================


class TestExtractTaxonomyCodes:
    """Unit tests for extract_taxonomy_codes."""

    def test_string_items_in_list_extracted(self) -> None:
        result = extract_taxonomy_codes(["BD-1800", "YF-3000"])
        assert result == {"bd-1800", "yf-3000"}

    def test_dict_items_with_code_key_extracted(self) -> None:
        result = extract_taxonomy_codes([{"code": "BD-1800"}])
        assert "bd-1800" in result

    def test_case_normalization_to_lowercase(self) -> None:
        result = extract_taxonomy_codes(["BD-1800"])
        assert "bd-1800" in result
        assert "BD-1800" not in result

    def test_none_input_returns_empty_set(self) -> None:
        assert extract_taxonomy_codes(None) == set()  # type: ignore[arg-type]

    def test_non_list_input_returns_empty_set(self) -> None:
        assert extract_taxonomy_codes("BD-1800") == set()  # type: ignore[arg-type]


# ===========================================================================
# taxonomy_utils — clean_services_rollup
# ===========================================================================


class TestCleanServicesRollup:
    """Tests for service rollup normalization."""

    def test_string_items_converted_to_service_objects_with_empty_taxonomies(self) -> None:
        """String items must be converted to service objects with an empty taxonomy list."""
        result = clean_services_rollup(["Food Pantry", "Shelter"])
        assert len(result) == 2
        names = {str(item["name"]) for item in result}
        assert "food pantry" in names
        assert "shelter" in names
        for item in result:
            assert item["taxonomies"] == []

    def test_dict_items_with_name_and_taxonomies_normalized(self) -> None:
        result = clean_services_rollup(
            [
                {"name": "Food Pantry", "taxonomies": [{"code": "BD-1800"}]},
            ]
        )
        assert len(result) == 1
        assert result[0]["name"] == "food pantry"
        taxonomies = result[0]["taxonomies"]
        assert isinstance(taxonomies, list)
        taxonomy_codes = [t.get("code") for t in taxonomies if isinstance(t, dict)]
        assert "bd-1800" in taxonomy_codes

    def test_service_taxonomy_codes_key_alias_recognized(self) -> None:
        """taxonomy_codes is a recognized alias for the taxonomies key."""
        result = clean_services_rollup(
            [
                {"name": "Service A", "taxonomy_codes": [{"code": "YF-3000"}]},
            ]
        )
        assert len(result) == 1
        taxonomies = result[0]["taxonomies"]
        assert isinstance(taxonomies, list)
        taxonomy_codes = [t.get("code") for t in taxonomies if isinstance(t, dict)]
        assert "yf-3000" in taxonomy_codes

    def test_service_rollup_preserves_service_and_taxonomy_identifiers(self) -> None:
        """Normalized service rollups must retain ids needed by cache persistence."""
        result = clean_services_rollup(
            [
                {
                    "id": "svc-1",
                    "name": "Service A",
                    "description": "Desc",
                    "taxonomy_codes": [
                        {"taxonomy_term_id": "tax-1", "code": "YF-3000", "name": "Food"}
                    ],
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["id"] == "svc-1"
        assert result[0]["description"] == "desc"
        taxonomies = result[0]["taxonomies"]
        assert isinstance(taxonomies, list)
        first_taxonomy = taxonomies[0]
        assert isinstance(first_taxonomy, dict)
        assert first_taxonomy["taxonomy_term_id"] == "tax-1"

    def test_empty_list_returns_empty(self) -> None:
        assert clean_services_rollup([]) == []

    def test_none_input_returns_empty(self) -> None:
        assert clean_services_rollup(None) == []  # type: ignore[arg-type]

    def test_non_dict_non_string_items_skipped(self) -> None:
        result = clean_services_rollup([42, None, {"name": "Valid Service", "taxonomies": []}])
        assert len(result) == 1
        assert result[0]["name"] == "valid service"

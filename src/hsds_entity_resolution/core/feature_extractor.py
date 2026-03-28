# pyright: reportMissingImports=false
# ruff: noqa
# fmt: off
import math
import difflib
import re
import json
import ast
import numpy as np
import os
from typing import Dict, Any, List, Set, Optional
import joblib

from hsds_entity_resolution.core.taxonomy_utils import (
    extract_entity_taxonomy_codes,
    taxonomy_hierarchy_levels,
)
try:
    __import__("sentence_transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class FeatureExtractor:
    """
    Extracts features from a pair of entities for deduplication models.
    """

    def __init__(
        self,
        rollup_services: bool = True,
        taxonomy_embeddings: Optional[Dict[str, List[float]]] = None,
        tfidf_vectorizer_path: Optional[str] = None,
        model_type: str = 'organization',
    ):
        """Initializes the FeatureExtractor.

        Args:
            rollup_services: Whether to include service data in organization features.
            taxonomy_embeddings: Pre-loaded taxonomy embeddings.
            tfidf_vectorizer_path: Path to a specific TF-IDF vectorizer model.
            model_type: The type of entity model ('organization' or 'service').
        """
        self.rollup_services = rollup_services
        self.taxonomy_embeddings = taxonomy_embeddings or {}
        self.model_type = model_type
        self.tfidf_vectorizer = None

        # Load TF-IDF vectorizer
        if tfidf_vectorizer_path and os.path.exists(tfidf_vectorizer_path):
            try:
                self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
                print(f"Loaded TF-IDF vectorizer from {tfidf_vectorizer_path}")
            except Exception as e:
                print(f"Error loading TF-IDF vectorizer: {e}")

        # Determine default path if not explicitly provided
        if self.tfidf_vectorizer is None:
            base_dir = os.path.dirname(__file__)
            default_path = os.path.join(
                base_dir,
                "LightGBM_Train",
                "tf_idf_models",
                f"tfidf_vectorizer_{model_type}.joblib",
            )
            legacy_path = os.path.join(base_dir, "models", "tfidf_vectorizer.joblib")

            if os.path.exists(default_path):
                try:
                    self.tfidf_vectorizer = joblib.load(default_path)
                    print(f"Loaded TF-IDF vectorizer from default path: {default_path}")
                except Exception as e:
                    print(f"Error loading TF-IDF vectorizer from default path: {e}")
            elif os.path.exists(legacy_path):
                try:
                    self.tfidf_vectorizer = joblib.load(legacy_path)
                    print(
                        f"Warning: Specific vectorizer not found. Loaded legacy TF-IDF vectorizer from: {legacy_path}"
                    )
                except Exception as e:
                    print(f"Error loading legacy TF-IDF vectorizer: {e}")
            else:
                print(f"Warning: No TF-IDF vectorizer found for {model_type}.")

        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Placeholder for loading sentence transformer model if needed
                pass
            except Exception:
                pass

    def extract_features(
        self, entity_a: Dict[str, Any], entity_b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extracts all features for a pair of entities.

        Args:
            entity_a: The first entity dictionary.
            entity_b: The second entity dictionary.

        Returns:
            A dictionary of extracted feature names and their float values.
        """
        features = {}
        features.update(self._data_completeness(entity_a, entity_b))
        features.update(self._token_completeness(entity_a, entity_b))
        features.update(self._geographic_features(entity_a, entity_b))
        features.update(self._string_similarity(entity_a, entity_b))

        # Taxonomy features are skipped for organization models as requested
        if self.model_type != 'organization':
            features.update(self._taxonomy_similarity(entity_a, entity_b))
            features.update(self._shared_taxonomy_extended_features(entity_a, entity_b))
            features.update(self._taxonomy_embedding_features(entity_a, entity_b))

        features.update(self._complexity_features(entity_a, entity_b))

        if self.model_type != 'organization':
            features.update(self._virtual_service_features(entity_a, entity_b))

        # Service aggregation features
        if self.rollup_services and ('services_rollup' in entity_a or 'services_rollup' in entity_b):
            service_features = self._service_aggregation_features(entity_a, entity_b)
            if self.model_type == 'organization':
                service_features.pop('min_avg_taxonomies_per_service', None)
            features.update(service_features)

        # Organization features (for services)
        if entity_a.get('organization_name') or entity_b.get('organization_name'):
            features.update(self._organization_features(entity_a, entity_b))

        # Enhanced features
        features.update(self._tfidf_features(entity_a, entity_b))
        features.update(self._ngram_features(entity_a, entity_b))
        features.update(self._statistical_features(entity_a, entity_b))

        return features

    def _get_location(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to get the first location from an entity."""
        locs = entity.get('locations', [])
        if isinstance(locs, list) and locs:
            return locs[0]
        return {}

    def _data_completeness(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes data completeness features."""
        loc_a = self._get_location(a)
        loc_b = self._get_location(b)

        return {
            'both_have_email': 1.0 if a.get('emails') and b.get('emails') else 0.0,
            'both_have_phone': 1.0 if a.get('phones') and b.get('phones') else 0.0,
            'both_have_website': 1.0 if a.get('websites') and b.get('websites') else 0.0,
            'both_have_cities': 1.0
            if loc_a.get('city') and loc_b.get('city')
            else 0.0,
            'both_have_states': 1.0
            if loc_a.get('state') and loc_b.get('state')
            else 0.0,
            'both_have_zip': 1.0
            if loc_a.get('postal_code') and loc_b.get('postal_code')
            else 0.0,
        }

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenizes text into a set of words."""
        if not text:
            return set()
        return set(re.findall(r'\w+', str(text).lower()))

    def _token_completeness(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes token completeness features."""
        name_a = str(a.get('name') or '')
        name_b = str(b.get('name') or '')

        tokens_a = self._tokenize(name_a)
        tokens_b = self._tokenize(name_b)

        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)

        jaccard = len(intersection) / len(union) if union else 0.0

        return {
            'token_jaccard': jaccard,
            'token_overlap_count': float(len(intersection)),
            'total_token_count': float(len(union)),
            'name_length_diff_in_tokens': float(abs(len(tokens_a) - len(tokens_b))),
        }

    def _geographic_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes geographic features."""
        loc_a = self._get_location(a)
        loc_b = self._get_location(b)

        city_a = str(loc_a.get('city') or '').lower().strip()
        city_b = str(loc_b.get('city') or '').lower().strip()

        state_a = str(loc_a.get('state') or '').lower().strip()
        state_b = str(loc_b.get('state') or '').lower().strip()

        zip_a = str(loc_a.get('postal_code') or '').split('-')[0].strip()
        zip_b = str(loc_b.get('postal_code') or '').split('-')[0].strip()

        return {
            'same_city': 1.0 if city_a and city_b and city_a == city_b else 0.0,
            'same_state': 1.0 if state_a and state_b and state_a == state_b else 0.0,
            'same_zipcode': 1.0 if zip_a and zip_b and zip_a == zip_b else 0.0,
        }

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculates Levenshtein ratio (0-1)."""
        if not s1 or not s2:
            return 0.0
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _token_sort_ratio(self, s1: str, s2: str) -> float:
        """Calculates Token Sort Ratio."""
        if not s1 or not s2:
            return 0.0

        tokens_a = sorted(re.findall(r'\w+', s1.lower()))
        tokens_b = sorted(re.findall(r'\w+', s2.lower()))

        sorted_a = " ".join(tokens_a)
        sorted_b = " ".join(tokens_b)

        return difflib.SequenceMatcher(None, sorted_a, sorted_b).ratio()

    def _jaro_winkler(self, s1: str, s2: str) -> float:
        """Calculates Jaro-Winkler distance."""
        if not s1 or not s2:
            return 0.0
        
        s1 = s1.lower()
        s2 = s2.lower()
        
        if s1 == s2:
            return 1.0
            
        len1, len2 = len(s1), len(s2)
        match_distance = (max(len1, len2) // 2) - 1
        
        matches = 0
        hash_s1 = [0] * len1
        hash_s2 = [0] * len2
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2[j] == s1[i] and hash_s2[j] == 0:
                    hash_s1[i] = 1
                    hash_s2[j] = 1
                    matches += 1
                    break
                    
        if matches == 0:
            return 0.0
            
        t = 0
        point = 0
        for i in range(len1):
            if hash_s1[i]:
                while hash_s2[point] == 0:
                    point += 1
                if s1[i] != s2[point]:
                    t += 1
                point += 1
        t /= 2
        
        jaro = ((matches / len1) + (matches / len2) + ((matches - t) / matches)) / 3.0
        
        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
                
        return jaro + (prefix * 0.1 * (1 - jaro))

    def _string_similarity(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes string similarity features."""
        name_a = str(a.get('name') or '')
        name_b = str(b.get('name') or '')

        return {
            'name_jaro_winkler': self._jaro_winkler(name_a, name_b),
            'name_levenshtein': self._levenshtein_ratio(name_a, name_b),
            'name_token_sort': self._token_sort_ratio(name_a, name_b),
        }

    def _taxonomy_similarity(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes taxonomy similarity features."""
        # Get taxonomy lists from both entities
        taxonomies_a = a.get('taxonomies', [])
        taxonomies_b = b.get('taxonomies', [])

        if not isinstance(taxonomies_a, list):
            taxonomies_a = []
        if not isinstance(taxonomies_b, list):
            taxonomies_b = []

        # Extract taxonomy codes
        codes_a = set()
        for tax in taxonomies_a:
            if isinstance(tax, dict) and tax.get('code'):
                codes_a.add(str(tax['code']))

        # Roll up taxonomies from services rollup for Entity A
        if self.rollup_services:
            services_a = a.get('services_rollup', [])
            if isinstance(services_a, list):
                for svc in services_a:
                    if isinstance(svc, dict) and svc.get('taxonomies'):
                        svc_taxs = svc['taxonomies']
                        if isinstance(svc_taxs, list):
                            for code in svc_taxs:
                                if code:
                                    codes_a.add(str(code))

        codes_b = set()
        for tax in taxonomies_b:
            if isinstance(tax, dict) and tax.get('code'):
                codes_b.add(str(tax['code']))

        # Roll up taxonomies from services rollup for Entity B
        if self.rollup_services:
            services_b = b.get('services_rollup', [])
            if isinstance(services_b, list):
                for svc in services_b:
                    if isinstance(svc, dict) and svc.get('taxonomies'):
                        svc_taxs = svc['taxonomies']
                        if isinstance(svc_taxs, list):
                            for code in svc_taxs:
                                if code:
                                    codes_b.add(str(code))

        count_a = len(codes_a)
        count_b = len(codes_b)

        shared_codes = codes_a.intersection(codes_b)
        shared_count = len(shared_codes)

        return {
            'shared_taxonomy_count': float(shared_count),
            'both_have_taxonomies': 1.0 if count_a > 0 and count_b > 0 else 0.0,
            'total_taxonomy_count': float(count_a + count_b),
            'taxonomy_count_diff': float(abs(count_a - count_b)),
        }

    def _service_aggregation_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes service aggregation features (organizations only)."""
        services_a = a.get('services_rollup', [])
        services_b = b.get('services_rollup', [])

        if not isinstance(services_a, list):
            services_a = []
        if not isinstance(services_b, list):
            services_b = []

        num_services_a = len(services_a)
        num_services_b = len(services_b)

        min_services = float(min(num_services_a, num_services_b))
        max_services = float(max(num_services_a, num_services_b))
        total_services = num_services_a + num_services_b
        service_count_diff = abs(num_services_a - num_services_b)
        both_have_services = 1.0 if num_services_a > 0 and num_services_b > 0 else 0.0

        if max_services > 0:
            service_count_ratio = min_services / max_services
        else:
            service_count_ratio = 0.0

        names_a = []
        for svc in services_a:
            if isinstance(svc, dict) and svc.get('name'):
                names_a.append(str(svc['name']).lower().strip())

        names_b = []
        for svc in services_b:
            if isinstance(svc, dict) and svc.get('name'):
                names_b.append(str(svc['name']).lower().strip())

        shared_service_names = 0
        matched_b = set()

        for name_a in names_a:
            for idx_b, name_b in enumerate(names_b):
                if idx_b not in matched_b:
                    similarity = self._jaro_winkler(name_a, name_b)
                    if similarity > 0.85:
                        shared_service_names += 1
                        matched_b.add(idx_b)
                        break

        service_name_jaccard = 0.0

        if names_a and names_b:
            tokens_a_list = [self._tokenize(name) for name in names_a]
            tokens_b_list = [self._tokenize(name) for name in names_b]

            for tokens_a in tokens_a_list:
                for tokens_b in tokens_b_list:
                    intersection = len(tokens_a.intersection(tokens_b))
                    union = len(tokens_a.union(tokens_b))
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > service_name_jaccard:
                            service_name_jaccard = jaccard
                            if service_name_jaccard == 1.0:
                                break
                if service_name_jaccard == 1.0:
                    break

        all_taxonomies_a = set()
        for svc in services_a:
            if isinstance(svc, dict) and svc.get('taxonomies'):
                taxonomies = svc['taxonomies']
                if isinstance(taxonomies, list):
                    for code in taxonomies:
                        if code:
                            all_taxonomies_a.add(str(code))

        all_taxonomies_b = set()
        for svc in services_b:
            if isinstance(svc, dict) and svc.get('taxonomies'):
                taxonomies = svc['taxonomies']
                if isinstance(taxonomies, list):
                    for code in taxonomies:
                        if code:
                            all_taxonomies_b.add(str(code))

        shared_taxonomies = all_taxonomies_a.intersection(all_taxonomies_b)
        shared_service_taxonomies = len(shared_taxonomies)

        taxonomy_union = all_taxonomies_a.union(all_taxonomies_b)
        total_unique_service_taxonomies = float(len(taxonomy_union))

        total_tax_count_a = 0
        for svc in services_a:
            if isinstance(svc, dict) and svc.get('taxonomies'):
                taxs = svc['taxonomies']
                if isinstance(taxs, list):
                    total_tax_count_a += len(taxs)

        total_tax_count_b = 0
        for svc in services_b:
            if isinstance(svc, dict) and svc.get('taxonomies'):
                taxs = svc['taxonomies']
                if isinstance(taxs, list):
                    total_tax_count_b += len(taxs)

        avg_taxonomies_per_service_a = (
            float(total_tax_count_a) / num_services_a if num_services_a > 0 else 0.0
        )
        avg_taxonomies_per_service_b = (
            float(total_tax_count_b) / num_services_b if num_services_b > 0 else 0.0
        )

        min_avg_tax = min(avg_taxonomies_per_service_a, avg_taxonomies_per_service_b)
        max_avg_tax = max(avg_taxonomies_per_service_a, avg_taxonomies_per_service_b)
        avg_tax_diff = abs(
            avg_taxonomies_per_service_a - avg_taxonomies_per_service_b
        )

        return {
            'min_services': min_services,
            'max_services': max_services,
            'total_services': float(total_services),
            'service_count_diff': float(service_count_diff),
            'service_count_ratio': service_count_ratio,
            'both_have_services': both_have_services,
            'shared_service_names': float(shared_service_names),
            'max_service_name_jaccard': service_name_jaccard,
            'shared_service_taxonomies': float(shared_service_taxonomies),
            'total_unique_service_taxonomies': total_unique_service_taxonomies,
            'min_avg_taxonomies_per_service': min_avg_tax,
            'max_avg_taxonomies_per_service': max_avg_tax,
            'avg_taxonomies_per_service_diff': avg_tax_diff,
        }

    def _organization_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes organization features (for services)."""
        name_a = str(a.get('organization_name') or '')
        name_b = str(b.get('organization_name') or '')
        id_a = str(a.get('organization_id') or '')
        id_b = str(b.get('organization_id') or '')

        if not name_a or not name_b:
            return {
                'org_name_jaro_winkler': 0.0,
                'org_name_levenshtein': 0.0,
                'org_name_token_sort': 0.0,
                'same_org_name_fuzzy': 0.0,
                'same_organization_id': 0.0,
            }
        
        # Calculate string similarities
        jaro = self._jaro_winkler(name_a, name_b)
        levenshtein = self._levenshtein_ratio(name_a, name_b)
        token_sort = self._token_sort_ratio(name_a, name_b)
        
        return {
            'org_name_jaro_winkler': jaro,
            'org_name_levenshtein': levenshtein,
            'org_name_token_sort': token_sort,
            'same_org_name_fuzzy': 1.0 if jaro > 0.85 else 0.0,
            'same_organization_id': 1.0 if id_a and id_b and id_a == id_b else 0.0,
        }

    def _complexity_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extracts complexity features (name/description length, sibling services)."""
        name_a = str(a.get('name') or '')
        name_b = str(b.get('name') or '')

        desc_a = str(a.get('description') or '')
        desc_b = str(b.get('description') or '')

        tokens_name_a = self._tokenize(name_a)
        tokens_name_b = self._tokenize(name_b)

        tokens_desc_a = self._tokenize(desc_a)
        tokens_desc_b = self._tokenize(desc_b)

        services_a = a.get('services_rollup') or []
        services_b = b.get('services_rollup') or []

        if not isinstance(services_a, list):
            services_a = []
        if not isinstance(services_b, list):
            services_b = []

        num_services_a = float(len(services_a))
        num_services_b = float(len(services_b))

        min_name_len = float(min(len(tokens_name_a), len(tokens_name_b)))
        max_name_len = float(max(len(tokens_name_a), len(tokens_name_b)))

        min_desc_len = float(min(len(tokens_desc_a), len(tokens_desc_b)))
        max_desc_len = float(max(len(tokens_desc_a), len(tokens_desc_b)))

        min_services = float(min(num_services_a, num_services_b))
        max_services = float(max(num_services_a, num_services_b))

        return {
            'name_complexity_min': min_name_len,
            'name_complexity_max': max_name_len,
            'name_complexity_diff': float(abs(len(tokens_name_a) - len(tokens_name_b))),
            'name_complexity_ratio': min_name_len / max_name_len
            if max_name_len > 0
            else 0.0,
            'description_complexity_min': min_desc_len,
            'description_complexity_max': max_desc_len,
            'description_complexity_diff': float(
                abs(len(tokens_desc_a) - len(tokens_desc_b))
            ),
            'description_complexity_ratio': min_desc_len / max_desc_len
            if max_desc_len > 0
            else 0.0,
            'num_services_min': min_services,
            'num_services_max': max_services,
            'num_services_diff': float(abs(num_services_a - num_services_b)),
            'num_services_ratio': min_services / max_services
            if max_services > 0
            else 0.0,
        }

    def _shared_taxonomy_extended_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extended shared taxonomy features including hierarchy matching and ratios."""
        codes_a = self._get_all_taxonomy_codes(a)
        codes_b = self._get_all_taxonomy_codes(b)

        total_unique = len(codes_a.union(codes_b))
        shared_count = len(codes_a.intersection(codes_b))

        percent_shared = 0.0
        if total_unique > 0:
            percent_shared = float(shared_count) / float(total_unique)

        hierarchy_match_score = 0.0
        if codes_a and codes_b:
            match_sum = 0.0
            comparisons = 0

            for ca in codes_a:
                best_match_for_ca = 0.0
                for cb in codes_b:
                    if ca == cb:
                        best_match_for_ca = 1.0
                        break

                    ca_clean = ca.strip().upper()
                    cb_clean = cb.strip().upper()

                    if ca_clean and cb_clean:
                        if ca_clean.startswith(cb_clean) or cb_clean.startswith(ca_clean):
                            curr_match = 0.5
                            if curr_match > best_match_for_ca:
                                best_match_for_ca = curr_match

                match_sum += best_match_for_ca
                comparisons += 1

            prefixes_a = self._extract_taxonomy_prefixes(codes_a)
            prefixes_b = self._extract_taxonomy_prefixes(codes_b)

            intersection_prefixes = len(prefixes_a.intersection(prefixes_b))
            union_prefixes = len(prefixes_a.union(prefixes_b))

            if union_prefixes > 0:
                hierarchy_match_score = float(intersection_prefixes) / float(union_prefixes)

        return {
            'percent_shared_taxonomies': percent_shared,
            'taxonomy_hierarchy_match_score': hierarchy_match_score,
        }

    def _taxonomy_embedding_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculates features based on pairwise cosine similarity of taxonomy embeddings."""
        zero_features = {
            'taxonomy_pairs_sim_gt_070': 0.0,
            'taxonomy_pairs_sim_gt_075': 0.0,
            'taxonomy_pairs_sim_gt_080': 0.0,
            'taxonomy_pairs_sim_gt_085': 0.0,
            'taxonomy_pairs_sim_gt_090': 0.0,
            'taxonomy_pairs_sim_gt_070_ratio': 0.0,
            'taxonomy_pairs_sim_gt_075_ratio': 0.0,
            'taxonomy_pairs_sim_gt_080_ratio': 0.0,
            'taxonomy_pairs_sim_gt_085_ratio': 0.0,
            'taxonomy_pairs_sim_gt_090_ratio': 0.0,
        }

        if not self.taxonomy_embeddings:
            return zero_features

        codes_a = self._get_all_taxonomy_codes(a)
        codes_b = self._get_all_taxonomy_codes(b)

        if not codes_a or not codes_b:
            return zero_features

        total_denom = float(len(codes_a) + len(codes_b))

        vecs_a = [
            self.taxonomy_embeddings[code]
            for code in codes_a
            if code in self.taxonomy_embeddings
        ]
        vecs_b = [
            self.taxonomy_embeddings[code]
            for code in codes_b
            if code in self.taxonomy_embeddings
        ]

        if not vecs_a or not vecs_b:
            return zero_features

        # Calculate pairwise similarities
        # Convert to numpy for efficiency if many taxonomies, but loops are fine for small N
        count_070 = 0
        count_075 = 0
        count_080 = 0
        count_085 = 0
        count_090 = 0
        
        for va in vecs_a:
            # Pre-calc norm A (or assume normalized? let's calculate to be safe)
            norm_a = math.sqrt(sum(x*x for x in va))
            if norm_a == 0: continue
                
            for vb in vecs_b:
                norm_b = math.sqrt(sum(x*x for x in vb))
                if norm_b == 0: continue
                
                dot = sum(x*y for x,y in zip(va, vb))
                sim = dot / (norm_a * norm_b)
                
                if sim > 0.70: count_070 += 1
                if sim > 0.75: count_075 += 1
                if sim > 0.80: count_080 += 1
                if sim > 0.85: count_085 += 1
                if sim > 0.90: count_090 += 1
                
        return {
            'taxonomy_pairs_sim_gt_070': float(count_070),
            'taxonomy_pairs_sim_gt_075': float(count_075),
            'taxonomy_pairs_sim_gt_080': float(count_080),
            'taxonomy_pairs_sim_gt_085': float(count_085),
            'taxonomy_pairs_sim_gt_090': float(count_090),
            # Ratios relative to total number of taxonomies involved
            'taxonomy_pairs_sim_gt_070_ratio': float(count_070) / total_denom if total_denom > 0 else 0.0,
            'taxonomy_pairs_sim_gt_075_ratio': float(count_075) / total_denom if total_denom > 0 else 0.0,
            'taxonomy_pairs_sim_gt_080_ratio': float(count_080) / total_denom if total_denom > 0 else 0.0,
            'taxonomy_pairs_sim_gt_085_ratio': float(count_085) / total_denom if total_denom > 0 else 0.0,
            'taxonomy_pairs_sim_gt_090_ratio': float(count_090) / total_denom if total_denom > 0 else 0.0,
        }

    def _get_all_taxonomy_codes(self, entity: Dict[str, Any]) -> Set[str]:
        """Helper to get all taxonomy codes for an entity (including service rollup)."""
        if self.rollup_services:
            return extract_entity_taxonomy_codes(entity=entity)
        return extract_entity_taxonomy_codes(
            entity={"taxonomies": entity.get("taxonomies", [])}
        )

    def _extract_taxonomy_prefixes(self, codes: Set[str]) -> Set[str]:
        """Extracts hierarchical prefixes from taxonomy codes."""
        prefixes = set()
        for code in codes:
            prefixes.update(taxonomy_hierarchy_levels(code))
        return prefixes

    def _virtual_service_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes virtual service features."""
        def is_virtual(entity: Dict[str, Any]) -> bool:
            locs = entity.get('locations', [])
            if not isinstance(locs, list) or not locs:
                return True

            has_physical_location = False
            for loc in locs:
                l_type = str(loc.get('location_type', '')).lower()
                if l_type != 'virtual':
                    has_physical_location = True
                    break

            return not has_physical_location

        is_virtual_a = is_virtual(a)
        is_virtual_b = is_virtual(b)

        return {
            'is_virtual_service_diff': 1.0 if is_virtual_a != is_virtual_b else 0.0,
            'both_are_virtual_services': 1.0 if is_virtual_a and is_virtual_b else 0.0,
        }

    def _tfidf_features(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
        """Computes TF-IDF based distinctive content features."""
        if not self.tfidf_vectorizer:
            return {'tfidf_weighted_similarity': 0.0, 'max_distinctive_term_ratio': 0.0}

        name_a = str(a.get('name') or '')
        desc_a = str(a.get('description') or '')
        text_a = f"{name_a} {desc_a}".strip()

        name_b = str(b.get('name') or '')
        desc_b = str(b.get('description') or '')
        text_b = f"{name_b} {desc_b}".strip()

        if not text_a or not text_b:
            return {'tfidf_weighted_similarity': 0.0, 'max_distinctive_term_ratio': 0.0}

        try:
            # Transform
            vectors = self.tfidf_vectorizer.transform([text_a, text_b])

            # Cosine similarity
            dense = vectors.toarray()
            norm_a = np.linalg.norm(dense[0])
            norm_b = np.linalg.norm(dense[1])

            sim = 0.0
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(dense[0], dense[1]) / (norm_a * norm_b)

            # Distinctive term ratio (overlap of top N terms)
            indices_a = vectors[0].nonzero()[1]
            indices_b = vectors[1].nonzero()[1]

            set_a = set(indices_a)
            set_b = set(indices_b)

            distinctive_a = 0.0
            if len(set_a) > 0:
                distinctive_a = len(set_a - set_b) / len(set_a)

            distinctive_b = 0.0
            if len(set_b) > 0:
                distinctive_b = len(set_b - set_a) / len(set_b)

            return {
                'tfidf_weighted_similarity': float(sim),
                'max_distinctive_term_ratio': float(max(distinctive_a, distinctive_b)),
            }
        except Exception:
            return {'tfidf_weighted_similarity': 0.0, 'max_distinctive_term_ratio': 0.0}

    def _ngram_features(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
        """Computes N-gram features (bigrams, trigrams)."""
        name_a = str(a.get('name') or '')
        name_b = str(b.get('name') or '')

        def get_ngrams(text, n):
            words = re.findall(r'\w+', str(text).lower())
            if len(words) < n:
                return set()
            return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

        bigrams_a = get_ngrams(name_a, 2)
        bigrams_b = get_ngrams(name_b, 2)

        trigrams_a = get_ngrams(name_a, 3)
        trigrams_b = get_ngrams(name_b, 3)

        def jaccard(s1, s2):
            if not s1 or not s2:
                return 0.0
            union = len(s1.union(s2))
            return len(s1.intersection(s2)) / union if union > 0 else 0.0

        return {
            'bigram_overlap': jaccard(bigrams_a, bigrams_b),
            'trigram_overlap': jaccard(trigrams_a, trigrams_b),
        }

    def _statistical_features(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> Dict[str, float]:
        """Computes statistical structural features."""
        desc_a = str(a.get('description') or '')
        desc_b = str(b.get('description') or '')

        len_a = len(desc_a)
        len_b = len(desc_b)

        ratio = 0.0
        if max(len_a, len_b) > 0:
            ratio = min(len_a, len_b) / max(len_a, len_b)

        # Token containment
        tokens_a = self._tokenize(desc_a)
        tokens_b = self._tokenize(desc_b)

        containment_a_in_b = 0.0
        if len(tokens_a) > 0:
            containment_a_in_b = len(tokens_a.intersection(tokens_b)) / len(tokens_a)

        containment_b_in_a = 0.0
        if len(tokens_b) > 0:
            containment_b_in_a = len(tokens_b.intersection(tokens_a)) / len(tokens_b)

        return {
            'description_length_ratio': ratio,
            'containment_asymmetry': abs(containment_a_in_b - containment_b_in_a),
        }


    def extract_features_from_pair(self, pair: Any) -> Dict[str, float]:
        """Extracts all features for a pair of entities, including signal-based features.

        Args:
            pair: A pair object containing entity_a, entity_b, and reasons.

        Returns:
            A dictionary of extracted features.
        """
        # Base features from entity dictionaries
        features = self.extract_features(pair.entity_a, pair.entity_b)

        # Extract features from DUPLICATE_REASONS (signals)
        for reason in pair.reasons:
            signal_name = reason.get('MATCH_TYPE', 'unknown')

            # Use raw contribution as the feature value
            raw_val = reason.get('RAW_CONTRIBUTION', 0.0)
            if isinstance(raw_val, float) and math.isnan(raw_val):
                raw_val = 0.0

            # Skip circular features
            if signal_name in ['lightgbm_model', 'lightgbm_model_similarity', 'ml_similarity']:
                continue

            features[signal_name] = raw_val

            # Calculate similarity score
            val_a = reason.get('ENTITY_A_VALUE')
            val_b = reason.get('ENTITY_B_VALUE')

            if val_a is not None and val_b is not None:
                sim_score = self._calculate_similarity(val_a, val_b)
            else:
                # Fallback to DB score if values missing
                sim_score = reason.get('SIMILARITY_SCORE', 0.0)
                if isinstance(sim_score, float) and math.isnan(sim_score):
                    sim_score = 0.0

            # We always include the similarity feature if the signal exists
            # EXCEPTION: For boolean shared_* variables, similarity is not useful
            boolean_signals = {
                'shared_email',
                'shared_phone',
                'shared_domain',
                'shared_address',
            }
            if signal_name not in boolean_signals:
                features[f"{signal_name}_similarity"] = sim_score

        # Add embedding similarity
        emb_sim = pair.embedding_similarity
        if (
            emb_sim is None
            or (isinstance(emb_sim, float) and math.isnan(emb_sim))
            or emb_sim == 0.0
        ):
            # Try to calculate from embedding vectors if available
            vec_a = pair.entity_a.get('embedding_vector')
            vec_b = pair.entity_b.get('embedding_vector')

            if (
                vec_a is not None
                and len(vec_a) > 0
                and vec_b is not None
                and len(vec_b) > 0
            ):
                emb_sim = self._calculate_embedding_similarity(vec_a, vec_b)
            else:
                emb_sim = 0.0

        features['embedding_similarity'] = emb_sim

        return features

    def _calculate_embedding_similarity(self, vec_a: Any, vec_b: Any) -> float:
        """Calculates cosine similarity between two embedding vectors."""
        try:
            # Parse if string
            if isinstance(vec_a, str):
                vec_a = json.loads(vec_a)
            if isinstance(vec_b, str):
                vec_b = json.loads(vec_b)

            # Convert to list of floats if needed
            if isinstance(vec_a, list) and isinstance(vec_b, list):
                # Simple dot product / norm calculation
                dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
                norm_a = math.sqrt(sum(a * a for a in vec_a))
                norm_b = math.sqrt(sum(b * b for b in vec_b))

                if norm_a > 0 and norm_b > 0:
                    return dot_product / (norm_a * norm_b)

            return 0.0
        except Exception:
            return 0.0

    def _calculate_similarity(self, val_a: Any, val_b: Any) -> float:
        """Calculates similarity between two values.

        If both are lists (or string-encoded lists), calculate Jaccard similarity.
        Otherwise, calculate string similarity ratio.
        """

        # Helper to parse potential list strings
        def parse_if_list(val):
            if isinstance(val, list):
                return val
            if (
                isinstance(val, str)
                and val.strip().startswith('[')
                and val.strip().endswith(']')
            ):
                try:
                    # Try parsing as JSON first (double quotes)
                    return json.loads(val)
                except Exception:
                    try:
                        # Try parsing as Python literal (single quotes)
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        pass
            return val

        val_a = parse_if_list(val_a)
        val_b = parse_if_list(val_b)

        # Handle list inputs (e.g. phones, emails)
        if isinstance(val_a, list) and isinstance(val_b, list):
            set_a = set(str(x) for x in val_a if x)
            set_b = set(str(x) for x in val_b if x)

            if not set_a or not set_b:
                return 0.0

            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))

            return intersection / union if union > 0 else 0.0

        # Handle scalar inputs
        str_a = str(val_a) if val_a is not None else ""
        str_b = str(val_b) if val_b is not None else ""

        if not str_a or not str_b:
            return 0.0

        return difflib.SequenceMatcher(None, str_a.lower(), str_b.lower()).ratio()


# fmt: on

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger('ml_assessment')

class StartupAssessmentModel:
    """
    Advanced model that processes startup data across multiple categories
    """

    def __init__(self, model_weights=None):
        self.weights = model_weights or {
            'financial': 0.3,
            'growth': 0.25,
            'team': 0.15,
            'product': 0.15,
            'market': 0.15
        }

    def assess_startup(self, startup_data: Dict[str, Any], human_override=None) -> Dict[str, Any]:
        features = self._define_features()
        extracted = {}
        cat_scores = {}
        for cat, catset in features.items():
            fvals = self._extract_feature_values(startup_data, catset)
            if human_override and cat in human_override:
                cat_scores[cat] = human_override[cat]
            else:
                cat_scores[cat] = self._score_category(fvals, catset)
            extracted[cat] = fvals
        overall = sum(cat_scores[c] * self.weights[c] for c in cat_scores)
        analysis = self._generate_analysis(cat_scores, extracted, startup_data)
        sp = self._calculate_success_probability(overall, cat_scores, startup_data)
        return {
            "overall_score": overall,
            "success_probability": sp,
            "category_scores": cat_scores,
            "analysis": analysis,
            "features": extracted
        }

    def _define_features(self):
        return {
            "financial": {
                "burn_rate": {
                    'weight': 0.15, 'transform': 'inverse', 'min': 10000, 'max': 1_000_000
                },
                "runway_months": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 24
                },
                "gross_margin": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 1
                },
                "ltv_cac_ratio": {
                    'weight': 0.25, 'transform': 'log', 'min': 0, 'max': 10
                },
                "monthly_revenue": {
                    'weight': 0.15, 'transform': 'log', 'min': 0, 'max': 1_000_000
                },
                "cac_payback_months": {
                    'weight': 0.10, 'transform': 'inverse', 'min': 1, 'max': 36
                }
            },
            "growth": {
                "user_growth_rate": {
                    'weight': 0.30, 'transform': 'linear', 'min': 0, 'max': 0.5
                },
                "revenue_growth_rate": {
                    'weight': 0.25, 'transform': 'linear', 'min': 0, 'max': 0.5
                },
                "churn_rate": {
                    'weight': 0.20, 'transform': 'inverse', 'min': 0, 'max': 0.2
                },
                "referral_rate": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 0.3
                },
                "activation_rate": {
                    'weight': 0.10, 'transform': 'linear', 'min': 0, 'max': 1
                }
            },
            "team": {
                "founder_domain_exp_yrs": {
                    'weight': 0.25, 'transform': 'log', 'min': 0, 'max': 20
                },
                "founder_exits": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 3
                },
                "team_completeness": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 1
                },
                "engineering_team_size": {
                    'weight': 0.15, 'transform': 'log', 'min': 1, 'max': 50
                },
                "team_experience_diversity": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 1
                }
            },
            "product": {
                "product_maturity_score": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 100
                },
                "nps_score": {
                    'weight': 0.20, 'transform': 'linear_shifted', 'min': -100, 'max': 100
                },
                "feature_adoption_rate": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 1
                },
                "technical_innovation_score": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 100
                },
                "dau_mau_ratio": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 1
                },
                "tech_debt_score": {
                    'weight': 0.15, 'transform': 'inverse', 'min': 0, 'max': 100
                }
            },
            "market": {
                "market_size": {
                    'weight': 0.20, 'transform': 'log', 'min': 1_000_000, 'max': 10_000_000_000
                },
                "market_growth_rate": {
                    'weight': 0.20, 'transform': 'linear', 'min': 0, 'max': 0.5
                },
                "market_share": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 0.3
                },
                "competitive_advantage_score": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 100
                },
                "regulatory_risk": {
                    'weight': 0.15, 'transform': 'inverse', 'min': 0, 'max': 1
                },
                "category_leadership_score": {
                    'weight': 0.15, 'transform': 'linear', 'min': 0, 'max': 100
                }
            }
        }

    def _extract_feature_values(self, data: Dict[str, Any], catset: Dict[str, Any]) -> Dict[str, float]:
        ret = {}
        for f, params in catset.items():
            val = data.get(f, 0)
            if f in ["gross_margin", "churn_rate", "user_growth_rate", "revenue_growth_rate", "market_growth_rate", "activation_rate", "feature_adoption_rate"]:
                if val > 1 and val <= 100:
                    val = val / 100
            ret[f] = val
        return ret

    def _score_category(self, fvals: Dict[str, float], catset: Dict[str, Any]) -> float:
        import numpy as np
        cscore = 0
        tw = 0
        for f, v in fvals.items():
            if f not in catset:
                continue
            p = catset[f]
            w = p['weight']
            norm_val = self._normalize_feature(v, p)
            cscore += norm_val * w
            tw += w
        if tw > 0:
            cscore = (cscore / tw) * 100
        return cscore

    def _normalize_feature(self, val: float, params: Dict[str, Any]) -> float:
        import numpy as np
        t = params['transform']
        mn = params['min']
        mx = params['max']
        cv = max(mn, min(val, mx))
        if t == "linear":
            return (cv - mn) / (mx - mn)
        elif t == "linear_shifted":
            return (cv - mn) / (mx - mn)
        elif t == "log":
            if cv <= 0:
                return 0
            lmin = np.log1p(mn) if mn > 0 else 0
            lmax = np.log1p(mx)
            lv = np.log1p(cv)
            return (lv - lmin) / (lmax - lmin)
        elif t == "inverse":
            rng = mx - mn
            if rng == 0:
                return 0.5
            return 1 - ((cv - mn) / rng)
        return 0

    def _generate_analysis(self, cat_scores: Dict[str, float], feats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {}
        ov = sum(cat_scores.values()) / len(cat_scores)
        if ov >= 75:
            analysis["overall"] = "Excellent overall performance"
        elif ov >= 60:
            analysis["overall"] = "Strong overall performance"
        elif ov >= 45:
            analysis["overall"] = "Moderate overall performance"
        else:
            analysis["overall"] = "Underperforming overall"

        for cat, sc in cat_scores.items():
            if cat == "financial":
                analysis[cat] = self._analyze_financial(sc, feats[cat], data)
            elif cat == "growth":
                analysis[cat] = self._analyze_growth(sc, feats[cat], data)
            elif cat == "team":
                analysis[cat] = self._analyze_team(sc, feats[cat], data)
            elif cat == "product":
                analysis[cat] = self._analyze_product(sc, feats[cat], data)
            elif cat == "market":
                analysis[cat] = self._analyze_market(sc, feats[cat], data)

        analysis["strengths"] = self._identify_strengths(cat_scores, feats)
        analysis["weaknesses"] = self._identify_weaknesses(cat_scores, feats)
        analysis["recommendations"] = self._generate_recommendations(cat_scores, feats, data)
        return analysis

    def _calculate_success_probability(self, overall_score, cat_scores, data):
        sp = overall_score
        stg = data.get("stage", "seed").lower()
        if stg in ["seed", "pre-seed"]:
            sp -= 5
        elif stg in ["series-b", "series-c", "growth"]:
            sp += 5
        return max(0, min(100, sp))

    def _analyze_financial(self, sc, feats, data) -> str:
        if sc >= 75:
            return "Excellent financial health with strong unit economics"
        elif sc >= 60:
            return "Good finances with sustainable business model"
        elif sc >= 45:
            if feats.get('ltv_cac_ratio', 0) < 0.5:
                return "Mediocre finances with borderline unit economics"
            else:
                return "Acceptable finances with areas for improvement"
        else:
            run = feats.get('runway_months', 0)
            if run < 6:
                return "Critical financial situation with short runway requiring immediate attention"
            else:
                return "Weak finances with risk of unsustainable burn rate"

    def _analyze_growth(self, sc, feats, data) -> str:
        if sc >= 75:
            return "Outstanding growth with strong traction metrics"
        elif sc >= 60:
            return "Healthy growth with balanced acquisition and retention"
        elif sc >= 45:
            c = feats.get('churn_rate', 0)
            if c > 0.1:
                return "Moderate growth with high churn hampering scaling potential"
            else:
                return "Moderate growth with opportunities for optimization"
        else:
            return "Struggling growth requiring fundamental acquisition and retention improvements"

    def _analyze_team(self, sc, feats, data) -> str:
        if sc >= 75:
            return "Highly experienced team with strong execution capabilities"
        elif sc >= 60:
            return "Good team with some experience gaps to address"
        elif sc >= 45:
            return "Moderate team with partial domain expertise or leadership shortfalls"
        else:
            return "Weak team lacking critical leadership or domain experience"

    def _analyze_product(self, sc, feats, data) -> str:
        if sc >= 75:
            return "Mature product with strong NPS and adoption indicating high user value"
        elif sc >= 60:
            return "Good product with some areas for improvement"
        elif sc >= 45:
            return "Average product lacking strong user engagement or differentiation"
        else:
            return "Underdeveloped product requiring significant improvements"

    def _analyze_market(self, sc, feats, data) -> str:
        if sc >= 75:
            return "Large addressable market with strong advantage and favorable conditions"
        elif sc >= 60:
            return "Good market opportunity with decent size and growth potential"
        elif sc >= 45:
            return "Middling market with smaller size or slower growth requiring close competition monitoring"
        else:
            return "Unfavorable market conditions - too small, saturated, or highly competitive"

    def _identify_strengths(self, cat_scores, feats) -> List[str]:
        s = []
        best_cat = max(cat_scores, key=lambda x: cat_scores[x])
        if cat_scores[best_cat] >= 70:
            s.append(f"Strong dimension: {best_cat}")
        return s

    def _identify_weaknesses(self, cat_scores, feats) -> List[str]:
        w = []
        worst_cat = min(cat_scores, key=lambda x: cat_scores[x])
        if cat_scores[worst_cat] <= 40:
            w.append(f"Weak dimension: {worst_cat}")
        return w

    def _generate_recommendations(self, cat_scores, feats, data) -> List[str]:
        recs = []
        if cat_scores.get('financial', 0) < 50:
            recs.append("Revisit financial strategy to reduce burn rate or improve revenue streams")
        if cat_scores.get('growth', 0) < 50:
            recs.append("Improve user acquisition and retention metrics by testing new channels")
        if cat_scores.get('team', 0) < 50:
            recs.append("Strengthen team by hiring for missing domain expertise or leadership")
        if cat_scores.get('product', 0) < 50:
            recs.append("Enhance product quality by gathering user feedback and fixing top friction points")
        if cat_scores.get('market', 0) < 50:
            recs.append("Consider strategic pivot as current market conditions may be challenging")
        if not recs:
            recs.append("Maintain momentum by scaling operations while sustaining quality metrics")
        return recs

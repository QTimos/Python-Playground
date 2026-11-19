import re
import unicodedata as ud
from typing import Literal, Dict
NormForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


class TextNormalizer:
    def __init__(self, lossy: bool) -> None:
        self.lossy = lossy
        self.whitespaces_map = {
            "WS": " ",
            "ZW": "",
            "NL": "\n"
            }
        self.whitespaces_pat = re.compile(
            "(?P<WS>[\x09\x0b\x0c\u0020\u00A0\u1680\u2000\u200A\u202F\u205F\u3000]+)|"
            "(?P<ZW>[\u200B\u200C\u200D\uFEFF]+)|"
            "(?P<NL>\r\n?)"
            )
        self.whitespaces_map_no_nl = {
            "WS": " ",
            "ZW": "",
            }
        self.whitespaces_pat_no_nl = re.compile(
            "(?P<WS>[\x09\x0b\x0c\u0020\u00A0\u1680\u2000\u200A\u202F\u205F\u3000]+)|"
            "(?P<ZW>[\u200B\u200C\u200D\uFEFF]+)"
            )
        self.collapse_patt = re.compile("(?P<SP> +)|(?P<NL>\n{3,})")
        self.collapse_map = {
                "SP": " ",
                "NL": "\n\n"
                }
        self.punct_patt = re.compile(
                r"(?P<QM>\?{2,})|(?P<EM>\!{2,})|(?P<HM>([—–]+)|-{2,})|(?P<DQ>[„“”])|(?P<SQ>[‘’`])"
                )
        self.punct_map = {
                "QM": "?",
                "EM": "!",
                "HM": "-",
                "DQ": "\"",
                "SQ": "'"
                }

    def _normalize_text(
            self,
            text: str,
            form: NormForm,
            ) -> str:
        """
        Returns the normal form form for the Unicode string 'text'.
        Valid values for form are ‘NFC’, ‘NFKC’, ‘NFD’, and ‘NFKD’.
        • NFD (Normal Form D): Decomposes characters into their base letters
            and separate combining marks (e.g., converting "Ç" into "C" + "◌̧").
        • NFC (Normal Form C): Decomposes and then recomposes characters
            into their single, pre-combined forms wherever possible
            (e.g., combining "C" + "◌̧" into "Ç").
        • NFKD (Normal Form KD): Decomposes compatibility characters into their
            standard equivalents and separates their combining marks
            (e.g., converting the Roman numeral "Ⅰ" into a standard "I").
        • NFKC (Normal Form KC): Decomposes compatibility characters into standard
            equivalents and then recomposes them into single, pre-combined forms.
        """
        if ud.is_normalized(form, text):
            return text
        return ud.normalize(form, text)

    def _handle_case(
            self,
            text: str,
            fold: bool = True
            ) -> str:
        """
        Returns a case-insensitively normalized text using
        Unicode's standard for casefolding.
        """
        if fold:
            return text.casefold()
        return text.lower()

    def _match(self, match: re.Match[str], map: Dict[str, str]) -> str:
        group_name = match.lastgroup if match.lastgroup else ""
        group_sub = map.get(group_name, None)
        if not (group_name == "" or group_sub is None):
            return group_sub
        return match.group()

    def _handle_punctuation(
            self,
            text: str,
            ) -> str:
        text = self.punct_patt.sub(
                lambda x : self._match(x, self.punct_map),
                text
                )
        return text

    def _handle_whitespaces(
            self,
            text: str,
            nl_normalize: bool = True,
            collapse: bool = True,
            strip: bool = True
            ) -> str:
        if nl_normalize:
            # Normalizing newlines to linux standard
            # Mac and Windows
            text = self.whitespaces_pat.sub(
                    lambda x : self._match(x, self.whitespaces_map),
                    text
                    )
        else:
            text = self.whitespaces_pat_no_nl.sub(
                    lambda x : self._match(x, self.whitespaces_map_no_nl),
                    text
                    )
        if collapse:
            text = self.collapse_patt.sub(
                    lambda x: self._match(x, self.collapse_map),
                    text
                    )
        if strip:
            text = text.strip()
        return text

    def _lossy_normalization(
            self,
            text: str
            ) -> str:
        normal = self._normalize_text(text, "NFKC")
        case_normal = self._handle_case(normal)
        punct_normal = self._handle_punctuation(case_normal)
        whitespace_normal = self._handle_whitespaces(punct_normal)
        return whitespace_normal

    def _lossless_normalization(
            self,
            text: str
            ) -> str:
        normal = self._normalize_text(text, "NFC")
        case_normal = self._handle_case(normal, fold=False)
        whitespace_normal = self._handle_whitespaces(
                case_normal,
                nl_normalize=False,
                collapse=False
                )
        return whitespace_normal

    def normalize(self, text: str) -> str:
        if self.lossy:
            return self._lossy_normalization(text)
        return self._lossless_normalization(text)







if __name__ == "__main__":
    text = "hello\r\r\nwhat is this ﬁ\nＩＢＭ\n\n"
    print("text: \"", text, "\"")
    normalizer = TextNormalizer(lossy=True)
    print("lossy: \"", normalizer.normalize(text), "\"")

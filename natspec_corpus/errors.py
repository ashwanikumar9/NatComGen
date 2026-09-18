"""Typed failures. Every one carries enough context to locate the input."""


class CorpusError(Exception):
    """Base. Never raised directly."""


class LexError(CorpusError):
    """The source could not be tokenised (unterminated string or block comment)."""


class ParseError(CorpusError):
    """A declaration was matched but could not be parsed."""


class InvariantError(CorpusError):
    """A post-build invariant failed. The build must not ship."""


class ResolutionError(CorpusError):
    """An @inheritdoc target could not be resolved."""

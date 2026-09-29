"""Function-level evaluation, scored the way the published systems score.

`tools/evaluate_all.py` answers "which configuration of ours is best". This
package answers a different question: "what number may be put next to
SmartDoc's 47.39". Those need different scopes and different conventions, and
mixing them is how a thesis acquires a table nobody can defend.

Three things are fixed here and nowhere else:

  scope        Functions only. Every system this work compares against
               documents functions; NatSpecGold does not.
  convention   Ba / B1..B4 in the LeClair reporting used by SmartDoc and the
               code-summarisation literature, kept separate from cumulative
               BLEU-1..4, which is a different statistic with a similar name.
  surface      One explicit normalisation, named in the output, applied to
               reference and hypothesis alike.
"""
__all__ = ["scope", "surface", "metrics"]

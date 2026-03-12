##
var text := 'Do not count your chickens before they hatch';
var words := text.ToWords;
var dict := words.Select(w -> w.Length)
  .Distinct
  .Order
  .Each(len -> words.Where(w -> w.Length = len));
dict.PrintLines;
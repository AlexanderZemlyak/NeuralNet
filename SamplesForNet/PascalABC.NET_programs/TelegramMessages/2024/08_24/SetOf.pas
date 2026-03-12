begin
  var s := SetOf(1,2,3);
  var s2 := SetOf(3,4,5);
  var symmDiff := (s - s2) + (s2 - s);
  Print(symmDiff);
end.
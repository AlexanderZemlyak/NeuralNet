begin
  var t := (1,2,4,8);
  foreach var x in t.ToArray do
    Print(x);
  Println;
  Print(t.ToArray[3]);
end.
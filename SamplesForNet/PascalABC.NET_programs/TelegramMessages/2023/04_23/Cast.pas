begin
  var L := new List<object>;
  L.Add(1);
  L.Add(2);
  L.Add(3.2);
  L.Add('abc');
  var q := L.Where(x-> x is integer).Cast&<integer>();
  Println(q);
  Println(q.Sum);
end.
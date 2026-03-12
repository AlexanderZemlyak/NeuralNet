begin
  var L := new List<object>;
  L.Add(1);
  L.Add(2);
  L.Add(3.2);
  L.Add('abc');
  var q := L.OfType&<integer>();
  Println(q);
  Println(q.Sum);
end.
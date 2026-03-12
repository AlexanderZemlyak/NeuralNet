begin
  var a := ArrRandomInteger(10,1,10);
  a.EachCount.Where(kv -> kv.Value > 1).Println
    .Select(kv -> kv.Key).Print
end.
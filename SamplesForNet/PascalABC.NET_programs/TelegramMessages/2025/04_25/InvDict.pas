begin
  var d := Dict('hello' to 'привет', 'dog' to 'собака', 'cat' to 'кошка');
  var dInv := Dict(d.Select(kv -> (kv.Value to kv.Key)));
  Print(dInv);
end.
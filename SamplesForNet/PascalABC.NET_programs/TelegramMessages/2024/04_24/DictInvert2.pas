begin
  var Capitals := Dict(('Россия','Москва'),('Франция','Париж'),('Китай','Пекин'));
  var Countries := Dict(Capitals.Select(kv->(kv.Value,kv.Key)));
  Capitals.Println;
  Countries.Println
end.
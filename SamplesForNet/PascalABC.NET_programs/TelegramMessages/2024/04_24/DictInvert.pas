begin
  var Capitals := Dict(('Россия','Москва'),('Франция','Париж'),('Китай','Пекин'));
  var Countries := Dict(Capitals.Values.ZipTuple(Capitals.Keys));
  Capitals.Println;
  Countries.Println
end.
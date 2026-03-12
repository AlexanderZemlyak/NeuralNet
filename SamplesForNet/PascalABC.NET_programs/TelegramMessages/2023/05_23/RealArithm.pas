begin
  var arithm := 1.0.Step(0.1).Take(11);
  arithm.Println;
  Print(arithm.Sum,arithm.Product.Round(3),arithm.Average);
end.
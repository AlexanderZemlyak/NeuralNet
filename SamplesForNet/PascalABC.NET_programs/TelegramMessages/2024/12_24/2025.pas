begin
  Arr(1..9).Select(x -> x * x * x).Sum.Println;
  Arr(1..9).Sum.Sqr.Println;
  '2025'.Batch(2).Select(x -> x[0]+x[1]).Select(x -> x.ToInteger).Sum.Sqr.Println;
  Seq(1,8,8,10).Select(x -> x ** 3).Sum.Round.Println;
  (2026..2029).Zip([4,-6,4,-1],(x,y) -> x * y).Sum.Println;
  (2026..2029).Select(x -> x*x).Zip([4,-6,4,-1],(x,y) -> x * y).Sum.Sqrt.Round.Println;
  (2021..2024).Reverse.Zip([4,-6,4,-1],(x,y) -> x * y).Sum.Println;
  (2021..2024).Reverse.Select(x -> x*x).Zip([4,-6,4,-1],(x,y) -> x * y).Sum.Sqrt.Round.Println;
end.
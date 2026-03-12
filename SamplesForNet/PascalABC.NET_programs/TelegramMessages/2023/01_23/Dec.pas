begin
  var d: decimal := 1;
  d := d / 3;
  Println(d);
  Println(1-d*3);
  Println(real(1-d*3));
  Println(decimal.MaxValue);
  Println(real(decimal.MaxValue));
  Println(SizeOf(decimal));
end.
var count, cnt: array['A'..'Z'] of integer;

begin
  Assign(input, '24.txt');
  
  while not eof do
  begin
    var s := ReadlnString;
    
    for var i := 2 to s.Length do
      if s[i - 1] = 'A' then 
        cnt[s[i]] += 1;
    
    var max := cnt.Max;
    
    for var c := 'A' to 'Z' do
      if cnt[c] = m then 
        count[c] += 1;
  end;
  
  Print(count.Max);
end.
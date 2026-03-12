const
  BorderSymbols = '─│┌┐└┘';

function BorderedText(text: string): string;
begin
  var (h,v,lu,ru,ld,rd) := BorderSymbols;
  var lines := text.Split('|');
  var maxlen := lines.Max(s->s.Length);
  var L := new List<string>;
  L.Add(lu+h*(maxlen+4)+ru);
  foreach var line in lines do
    L.Add(v+2*' '+line.PadRight(maxlen)+2*' '+v);
  L.Add(ld+h*(maxlen+4)+rd);
  Result := L.JoinToString(NewLine);
end;

begin
  var text := 'if условие then|  оператор1|else оператор2';
  Println(BorderedText(text));
end.
// https://rosettacode.org/wiki/Strip_comments_from_a_string#PascalABC.NET

function RemoveComments(s,delim: string): string
  := Regex.Replace(s, delim + '.+', '');

begin
  Writeln(RemoveComments('apples, pears # and bananas','#'));
  Writeln(RemoveComments('apples, pears ; and bananas'';'));
end.
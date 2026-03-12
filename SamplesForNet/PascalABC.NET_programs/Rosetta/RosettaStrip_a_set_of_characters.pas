// https://rosettacode.org/wiki/Strip_a_set_of_characters_from_a_string#PascalABC.NET

function StripChars(s,chars: string): string
  := s.Where(c -> c not in chars).JoinToString;

begin
  Print(StripChars('She was a soul stripper. She took my heart!','aei'));
end.
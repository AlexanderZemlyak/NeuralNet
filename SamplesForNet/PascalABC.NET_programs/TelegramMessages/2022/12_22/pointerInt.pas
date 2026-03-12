uses System.Runtime.InteropServices;

type 
[StructLayout(LayoutKind.Explicit)] 
Rec = record
  [FieldOffset(0)] i: integer;
  [FieldOffset(0)] r: real;
  [FieldOffset(0)] b: boolean;
  [FieldOffset(0)] pi: pinteger;
  [FieldOffset(0)] pr: preal;
  [FieldOffset(0)] pb: pboolean;
end;

begin
  var r: Rec;
  Println('Размер типа:',sizeof(Rec));
  r.i := 77;
  Print(r.i);
  r.r := 3.14;
  Print(r.r);
end.
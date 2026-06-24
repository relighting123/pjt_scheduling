export interface ShortCodeMap {
  codeByKey: Record<string, string>;
  ordered: { code: string; key: string }[];
}

/** 긴 PPK/OPER 키를 P1, O1 형태의 짧은 코드로 매핑 */
export function buildShortCodeMap(keys: string[], prefix: string): ShortCodeMap {
  const sorted = [...new Set(keys)].sort();
  const codeByKey: Record<string, string> = {};
  const ordered = sorted.map((key, i) => {
    const code = `${prefix}${i + 1}`;
    codeByKey[key] = code;
    return { code, key };
  });
  return { codeByKey, ordered };
}

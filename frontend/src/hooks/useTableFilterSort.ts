import { useMemo, useState } from "react";

export type SortDir = "asc" | "desc";

export interface SortState {
  key: string;
  dir: SortDir;
}

export function useTableFilterSort<T>(
  rows: T[],
  filterFn: (row: T, query: string) => boolean,
  sortFn: (a: T, b: T, key: string, dir: SortDir) => number,
  defaultSort: SortState,
) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortState>(defaultSort);

  const toggleSort = (key: string) => {
    setSort((prev) => (
      prev.key === key
        ? { key, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "asc" }
    ));
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q ? rows.filter((row) => filterFn(row, q)) : [...rows];
    list.sort((a, b) => sortFn(a, b, sort.key, sort.dir));
    return list;
  }, [rows, query, sort, filterFn, sortFn]);

  return { query, setQuery, sort, toggleSort, filtered };
}

export function compareStrings(a: string, b: string, dir: SortDir): number {
  const cmp = a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
  return dir === "asc" ? cmp : -cmp;
}

export function compareNumbers(a: number, b: number, dir: SortDir): number {
  const cmp = a - b;
  return dir === "asc" ? cmp : -cmp;
}

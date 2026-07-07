/** 테이블 데이터 → Excel(.xls) 다운로드. 별도 라이브러리 없이 HTML 테이블을
 * application/vnd.ms-excel MIME으로 저장하는 방식(엑셀이 그대로 워크시트로 인식). */

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function triggerDownload(filename: string, blob: Blob): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** headers: 컬럼 헤더 문자열 배열. rows: 각 행의 셀 값 배열(헤더와 순서 동일). */
export function downloadExcel(
  filename: string,
  headers: string[],
  rows: (string | number | boolean | null | undefined)[][],
): void {
  const headHtml = `<tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
  const bodyHtml = rows
    .map((r) => `<tr>${r.map((c) => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`)
    .join("");
  const html =
    `<html xmlns:o="urn:schemas-microsoft-com:office:office" ` +
    `xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">` +
    `<head><meta charset="UTF-8"></head>` +
    `<body><table border="1">${headHtml}${bodyHtml}</table></body></html>`;
  triggerDownload(filename, new Blob(["﻿", html], { type: "application/vnd.ms-excel;charset=utf-8" }));
}

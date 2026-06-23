interface DatasetEmptyPanelProps {
  loadError: string | null;
  onGoToDataset: () => void;
}

export default function DatasetEmptyPanel({
  loadError,
  onGoToDataset,
}: DatasetEmptyPanelProps) {
  return (
    <section className="empty-dataset card card-stagger">
      <h2>데이터셋이 없습니다</h2>
      <p className="empty-dataset-lead">
        dataset 폴더가 비어 있거나 JSON이 없습니다.
        외부에서 데이터를 준비한 뒤 API 서버를 재시작하거나 데이터를 새로고침하세요.
      </p>
      {loadError && <p className="empty-dataset-error">{loadError}</p>}
      <div className="empty-dataset-actions">
        <button type="button" className="btn btn-secondary" onClick={onGoToDataset}>
          데이터셋 조회로 이동
        </button>
      </div>
    </section>
  );
}

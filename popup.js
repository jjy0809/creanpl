document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('toggle');

  // 초기 상태 로드
  chrome.storage.local.get('isActive', ({ isActive }) => {
    toggle.checked = !!isActive;
  });

  // 토글 변경 이벤트
  toggle.addEventListener('change', (e) => {
    chrome.storage.local.set({ isActive: e.target.checked });
  });
});

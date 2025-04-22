// 아이콘 경로 설정
const ICONS = {
    on: {
    16: "icons/icon_on.png",
    32: "icons/icon_on.png",
    48: "icons/icon_on.png",
    128: "icons/icon_on.png"
    },
    off: {
    16: "icons/icon_off.png",
    32: "icons/icon_off.png",
    48: "icons/icon_off.png",
    128: "icons/icon_off.png"
    }
};

  // 아이콘 업데이트 함수
async function updateIcon(isActive) {
    const iconSet = isActive ? ICONS.on : ICONS.off;
    await chrome.action.setIcon({ path: iconSet });
}

chrome.storage.onChanged.addListener((changes) => {
    if (changes.isActive) {
    const isActive = changes.isActive.newValue;
    chrome.action.setIcon({ path: isActive ? ICONS.on : ICONS.off });
    }
});

  // 확장 프로그램 설치 시 초기화
chrome.runtime.onInstalled.addListener(async () => {
    const { isActive } = await chrome.storage.local.get('isActive');
    await updateIcon(!!isActive);
});

  // 상태 변경 감지
chrome.storage.onChanged.addListener((changes) => {
    if (changes.isActive) {
    updateIcon(changes.isActive.newValue);
    }
});

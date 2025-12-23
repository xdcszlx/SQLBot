<script lang="ts" setup>
import aboutBg from '@/assets/embedded/LOGO-about.png'

import { ref, onMounted } from 'vue'
import { licenseApi } from '@/api/license'
import { useI18n } from 'vue-i18n'
const dialogVisible = ref(false)
const { t } = useI18n()
const build = ref('')

onMounted(() => {
  initVersion()
})

const initVersion = () => {
  licenseApi.version().then((res) => {
    build.value = res
  })
}

const open = () => {
  dialogVisible.value = true
}

defineExpose({
  open,
})
</script>

<template>
  <el-dialog
    v-model="dialogVisible"
    :title="t('about.title')"
    width="840px"
    modal-class="about-dialog"
  >
    <div class="color-overlay flex-center">
      <img width="368" height="84" :src="aboutBg" />
    </div>
    <div class="content">
      <div class="quote-section">
        <div class="quote-icon">"</div>
        <div class="quote-text">
          <p class="main-quote">
            数据如同璀璨星辰，散落于浩瀚夜空。
          </p>
          <p class="main-quote">
            智能则是那穿越时空的光，照亮未知，指引方向。
          </p>
          <p class="sub-quote">
            愿每一次探索，都能让洞察更深刻；
          </p>
          <p class="sub-quote">
            愿每一次决策，都能让未来更清晰。
          </p>
          <p class="dedication">
            —— 献给每一位探索数据奥秘的使用者
          </p>
        </div>
      </div>
      
      <div class="version-info">
        <div class="item">
          <div class="label">{{ $t('about.version_num') }}</div>
          <div class="value">{{ build }}</div>
        </div>
      </div>
    </div>
    <div class="name">作者：63726部队技术室     周立新   @18169583303</div>
  </el-dialog>
</template>

<style lang="less">
.about-dialog {
  .color-overlay {
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    background: var(--ed-color-primary-1a, #1cba901a);
    border: 1px solid #dee0e3;
    border-bottom: 0;
    height: 180px;
  }

  .name {
    font-weight: 400;
    font-size: 12px;
    line-height: 22px;
    text-align: center;
    margin-top: 16px;
    color: #8f959e;
  }

  .content {
    border-radius: 6px;
    border: 1px solid #dee0e3;
    border-top-left-radius: 0;
    border-top-right-radius: 0;
    border-top: 0;
    padding: 40px;

    .quote-section {
      position: relative;
      padding: 32px 24px;
      background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
      border-radius: 12px;
      margin-bottom: 32px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);

      .quote-icon {
        position: absolute;
        top: -10px;
        left: 20px;
        font-size: 72px;
        font-weight: bold;
        color: var(--ed-color-primary, #1cba90);
        opacity: 0.2;
        line-height: 1;
      }

      .quote-text {
        position: relative;
        z-index: 1;
        
        p {
          margin: 0;
          padding: 0;
        }

        .main-quote {
          font-size: 18px;
          line-height: 32px;
          color: #1f2329;
          font-weight: 500;
          margin-bottom: 12px;
          text-align: center;
        }

        .sub-quote {
          font-size: 16px;
          line-height: 28px;
          color: #646a73;
          margin-bottom: 8px;
          text-align: center;
          font-style: italic;
        }

        .dedication {
          font-size: 14px;
          line-height: 24px;
          color: #8f959e;
          margin-top: 24px;
          text-align: right;
          font-style: italic;
        }
      }
    }

    .version-info {
      padding: 16px 24px;
      background: #f8f9fa;
      border-radius: 8px;

      .item {
        font-size: 14px;
        font-style: normal;
        font-weight: 400;
        line-height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;

        .label {
          color: #646a73;
          margin-right: 12px;
        }

        .value {
          color: #1f2329;
          font-weight: 500;
        }
      }
    }
  }
}
</style>

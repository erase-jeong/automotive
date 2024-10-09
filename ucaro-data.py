import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

plt.matplotlib.use('Agg')

# Heroku 포트 설정 추가
port = os.environ.get("PORT", 8501)

st.title('유카로 고객추천시스템')


# 사이드바에서 데이터 분류 선택
option = st.sidebar.radio("카테고리", ["아우디", "폭스바겐"], key=3)

if option == "아우디":
    

    st.header("<아우디>")
    st.text("") 
    st.header("지점 분포현황")
    st.write("- 남천 1, 남천 2는 남천으로, 해운대1, 해운대2는 해운대로 통합 \n- 선등록, AAP, HQ, MAL 라고 되어있는 지점데이터는 제외")

    # 엑셀 파일 경로 설정
    file_path = 'data/audi_store.xlsx'
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)


    #font_path = os.path.join("data", "MALGUN.TTF")  # 데이터 폴더 내의 폰트 파일 경로

    font_path = 'font/MALGUN.TTF'  # 폰트 경로 설정
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)



    # 2. 지점명 통합 (남천1, 남천2는 남천으로 / 해운대1, 해운대2는 해운대로)
    df['지점명'] = df['지점명'].replace({
        '남천1': '남천', '남천2': '남천',
        '해운대1': '해운대', '해운대2': '해운대'
    })

    # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
    df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL'])]

    # 4. 지점별로 개수 카운트
    branch_count = df['지점명'].value_counts()

    # 데이터프레임으로 변환
    branch_count_df = branch_count.reset_index()
    branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
    # 5. 전체 고객 수 계산
    total_customers = branch_count.sum()


    # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
    #st.subheader('지점별 분포현황')

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

    # 막대 위에 비율 표시
    for index, value in enumerate(branch_count.values):
        percentage = (value / total_customers) * 100
        ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

    ax.set_title('지점별 분포현황')
    ax.set_xlabel('지점명')
    ax.set_ylabel('판매 개수')
    ax.set_xticks(range(len(branch_count)))
    ax.set_xticklabels(branch_count.index, rotation=45)

    # 그래프 Streamlit에 표시
    st.pyplot(fig)

    # Streamlit에서 데이터프레임 표시
    #st.subheader('지점별 판매 개수')

    st.dataframe(branch_count_df, use_container_width=True)

    st.divider() 


    st.header("지점별 고객 거주지 분포현황")




    # 지점 선택을 위한 드롭다운 메뉴
    selected_branch = st.selectbox('지점을 선택하세요:', df['지점명'].unique())

    # 선택된 지점에 해당하는 데이터 필터링
    selected_df = df[df['지점명'] == selected_branch]

    # 주소에서 첫 번째 부분(도시 또는 시, 구 등) 추출
    selected_df['거주지'] = selected_df['주소'].apply(lambda x: str(x).split()[0])

    # 거주지별로 카운트
    address_count = selected_df['거주지'].value_counts()

    # 선택된 지점 거주지 분포 데이터 정리
    address_count_df = pd.DataFrame(address_count).reset_index()
    address_count_df.columns = ['거주지', '카운트']

    # 전체 고객 수 계산
    total_customers = address_count_df['카운트'].sum()

    # 각 거주지별 비율 계산
    address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100


    # 시각화 (막대 그래프) - 비율 표시 추가
    st.subheader(f'{selected_branch} 지점 고객 거주지 분포 및 비율')

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

    # 막대 위에 비율 표시
    for index, value in enumerate(address_count_df['카운트']):
        ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

    ax.set_title(f'{selected_branch} 지점 고객 거주지 분포 및 비율')
    ax.set_xlabel('거주지')
    ax.set_ylabel('고객 수')
    ax.set_xticks(range(len(address_count_df)))
    ax.set_xticklabels(address_count_df['거주지'], rotation=45)

    # 그래프를 Streamlit에 표시
    st.pyplot(fig)

    # Streamlit에서 데이터프레임 표시
    st.dataframe(address_count_df, use_container_width=True)

elif option == "폭스바겐":

    st.header("<폭스바겐>")
    st.text("") 
    st.header("지점 분포현황")
    st.write("- 창원1, 창원2 지점 창원으로 통합\n - 선등록, AAP, HQ, MAL, VMA 제외")

    # 엑셀 파일 경로 설정
    file_path = 'data/vx_store.xlsx'
    # 엑셀 파일 읽기
    df = pd.read_excel(file_path)

    #font_path = os.path.join("data", "MALGUN.TTF")

    font_path = 'font/MALGUN.TTF'  # 폰트 경로 설정
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)


    df['지점명'] = df['지점명'].replace({
    '창원1': '창원', '창원2': '창원'
    })

    # 3. 제외할 지점 제거 (선등록, AAP, HQ, MAL 제거)
    df = df[~df['지점명'].isin(['선등록', 'AAP', 'HQ', 'MAL','VWA'])]

    # 4. 지점별로 개수 카운트
    branch_count = df['지점명'].value_counts()

    # 데이터프레임으로 변환
    branch_count_df = branch_count.reset_index()
    branch_count_df.columns = ['지점명', '고객 수']  # 열 이름 변경
    # 5. 전체 고객 수 계산
    total_customers = branch_count.sum()


    # 6. 결과 시각화 (막대 그래프) - 비율 표시 추가
    #st.subheader('지점별 분포현황')

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=branch_count.index, y=branch_count.values, palette="Blues_d", ax=ax)

    # 막대 위에 비율 표시
    for index, value in enumerate(branch_count.values):
        percentage = (value / total_customers) * 100
        ax.text(index, value + 10, f"{percentage:.2f}%", ha='center')

    plt.title('지점별 분포현황')
    plt.xlabel('지점명')
    plt.ylabel('판매 개수')
    ax.set_xticks(range(len(branch_count)))
    ax.set_xticklabels(branch_count.index, rotation=45)

    # 그래프 Streamlit에 표시
    st.pyplot(fig)

    # Streamlit에서 데이터프레임 표시
    #st.subheader('지점별 판매 개수')

    st.dataframe(branch_count_df, use_container_width=True)

    st.divider() 


    st.header("지점별 고객 거주지 분포현황")




    # 지점 선택을 위한 드롭다운 메뉴
    selected_branch = st.selectbox('지점을 선택하세요:', df['지점명'].unique())

    # 선택된 지점에 해당하는 데이터 필터링
    selected_df = df[df['지점명'] == selected_branch]

    # 주소에서 첫 번째 부분(도시 또는 시, 구 등) 추출
    selected_df['거주지'] = selected_df['주소'].apply(lambda x: str(x).split()[0])

    # 거주지별로 카운트
    address_count = selected_df['거주지'].value_counts()

    # 선택된 지점 거주지 분포 데이터 정리
    address_count_df = pd.DataFrame(address_count).reset_index()
    address_count_df.columns = ['거주지', '카운트']

    # 전체 고객 수 계산
    total_customers = address_count_df['카운트'].sum()

    # 각 거주지별 비율 계산
    address_count_df['비율 (%)'] = (address_count_df['카운트'] / total_customers) * 100


    # 시각화 (막대 그래프) - 비율 표시 추가
    st.subheader(f'{selected_branch} 지점 고객 거주지 분포 및 비율')

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=address_count_df['거주지'], y=address_count_df['카운트'], palette="Blues_d", ax=ax)

    # 막대 위에 비율 표시
    for index, value in enumerate(address_count_df['카운트']):
        ax.text(index, value + 10, f"{address_count_df['비율 (%)'].iloc[index]:.2f}%", ha='center')

    plt.title(f'{selected_branch} 지점 고객 거주지 분포 및 비율')
    plt.xlabel('거주지')
    plt.ylabel('고객 수')
    ax.set_xticks(range(len(address_count_df)))
    ax.set_xticklabels(address_count_df['거주지'], rotation=45)

    # 그래프를 Streamlit에 표시
    st.pyplot(fig)

    # Streamlit에서 데이터프레임 표시
    st.dataframe(address_count_df, use_container_width=True)





